"""Discovery ledger (spec §3.10, §9.7): candidates found by searches, their pre-declared assay
plans, results, status history and evidence level. Runs inside the controller (single DB
writer); assay checks run in the controller's process pool.

Statuses: new → triaged (manual) → validated / rejected / inconclusive (only from assay
results, by the plan's pre-declared rules). ``assayed`` is reserved for plans without an
automatic rule. Evidence: E1 at registration (measured by the search objective), E2 when every
check of the plan passes. Higher levels are never assigned automatically.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import TYPE_CHECKING, Any

from alife_lab.core.hashing import canonical_json, sha256_json
from alife_lab.core.spec import ExperimentSpec, SearchExperiment
from alife_lab.evolution import assays

if TYPE_CHECKING:
    from alife_lab.service.controller import Controller

STATUSES = ("new", "triaged", "assayed", "validated", "rejected", "inconclusive")
MANUAL_STATUSES = ("new", "triaged")
LIMITATIONS = [
    "Evidence from one objective (lenia_moving_localized v1) and one substrate implementation.",
    "The centroid speed is not a creature tracker; separate blobs share one centroid.",
    "No intervention (E3) or independent-implementation (E4) checks are part of this plan.",
]


class DiscoveryService:
    def __init__(self, controller: Controller) -> None:
        self.c = controller
        self.tasks: dict[str, asyncio.Task[None]] = {}
        # Assays interrupted by a controller stop never publish partial verdicts.
        self.c.db.execute(
            "UPDATE assays SET state='interrupted', finished_at=? WHERE state='running'",
            (time.time(),),
        )

    # ------------------------------------------------------------------ helpers

    def _event(self, did: str, kind: str, author: str, payload: dict[str, Any]) -> None:
        self.c.db.execute(
            "INSERT INTO discovery_events(discovery_id, at, kind, author, payload_json) "
            "VALUES (?,?,?,?,?)",
            (did, time.time(), kind, author, canonical_json(payload)),
        )

    def _row(self, did: str) -> dict[str, Any]:
        from alife_lab.service.controller import NotFound

        row = self.c.db.one("SELECT * FROM discoveries WHERE id=?", (did,))
        if row is None:
            raise NotFound(f"discovery {did!r} not found")
        return dict(row)

    def _set_status(self, did: str, status: str, level: str | None, author: str, why: str) -> None:
        row = self._row(did)
        level = level or row["evidence_level"]
        self.c.db.execute(
            "UPDATE discoveries SET status=?, evidence_level=?, updated_at=? WHERE id=?",
            (status, level, time.time(), did),
        )
        change = {"from": row["status"], "to": status, "evidence_level": level, "reason": why}
        self._event(did, "status_changed", author, change)

    # ------------------------------------------------------------------ lifecycle

    async def create(
        self,
        run_id: str,
        candidate: int,
        claim_kind: str,
        title: str,
        tick: int | None = None,
        author: str = "local_user",
    ) -> dict[str, Any]:
        from alife_lab.service.controller import ControllerError

        if claim_kind not in assays.CLAIM_KINDS:
            raise ControllerError(f"unknown claim kind {claim_kind!r}")
        title = title.strip()[:200] or f"{claim_kind} candidate {candidate}"
        run = self.c.run_summary(run_id)
        spec = ExperimentSpec.model_validate(run["spec"])
        search = spec.experiment
        if not isinstance(search, SearchExperiment):
            raise ControllerError("discoveries are registered from search runs")
        if search.target.substrate != "lenia" or search.asal is not None:
            raise ControllerError(
                "assay plan v1 covers single-channel Lenia candidates of non-CLIP searches only "
                "(a prompt-specificity assay for CLIP searches is planned)"
            )
        source_tick = int(run["clock"]["value"]) if tick is None else int(tick)
        info = await self.c.inspect(run_id, {"candidate": int(candidate)}, source_tick)
        cand = info.get("candidate")
        if not cand or cand.get("spec") is None:
            raise ControllerError("candidate has no valid target spec")
        cand_spec = ExperimentSpec.model_validate(cand["spec"])
        recorded = {k: cand[k] for k in ("score", "speed", "mass", "alive_fraction",
                                          "viable_fraction") if k in cand}  # fmt: skip
        plan = assays.make_plan(search, cand_spec.model_dump(mode="json"), recorded, claim_kind)
        did = "dsc_" + secrets.token_hex(8)
        now = time.time()
        source_ref = {
            "run_id": run_id,
            "tick": source_tick,
            "candidate_index": int(candidate),
            "generation": cand.get("generation"),
            "params": cand.get("params"),
            "lineage": cand.get("lineage", []),
            "search_spec_hash": spec.digest(),
            "objective": search.dynamics.objective,
            "algorithm": search.dynamics.algorithm,
        }
        self.c.db.execute(
            "INSERT INTO discoveries(id, title, claim_kind, substrate, source_run_id, "
            "source_tick, source_ref_json, candidate_spec_json, candidate_spec_hash, "
            "recorded_json, status, evidence_level, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, title, claim_kind, cand_spec.experiment.substrate, run_id, source_tick,
             canonical_json(source_ref), canonical_json(cand_spec.model_dump(mode="json")),
             cand_spec.digest(), canonical_json(recorded), "new", "E1", now, now),
        )  # fmt: skip
        aid = "asy_" + secrets.token_hex(8)
        self.c.db.execute(
            "INSERT INTO assays(id, discovery_id, plan_json, plan_hash, state, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (aid, did, canonical_json(plan), plan["plan_hash"], "planned", now),
        )
        self._event(did, "registered", author, {"source": source_ref, "claim_kind": claim_kind,
                                                 "plan_hash": plan["plan_hash"]})  # fmt: skip
        self.c._emit_event(run_id, "discovery_registered", {"discovery_id": did})
        return self.get(did)

    def review(
        self, did: str, status: str | None, note: str | None, author: str = "local_user"
    ) -> dict[str, Any]:
        from alife_lab.service.controller import ControllerError

        self._row(did)
        if status is not None:
            if status not in MANUAL_STATUSES:
                raise ControllerError(
                    f"status {status!r} can only be set by assay results (manual: "
                    f"{', '.join(MANUAL_STATUSES)})"
                )
            self._set_status(did, status, None, author, "manual review")
        if note:
            self._event(did, "note", author, {"text": note.strip()[:2000]})
        return self.get(did)

    async def start_assay(self, did: str, author: str = "local_user") -> dict[str, Any]:
        from alife_lab.service.controller import Conflict, ControllerError

        self._row(did)
        if did in self.tasks and not self.tasks[did].done():
            raise Conflict("an assay is already running for this discovery")
        plan_row = self.c.db.one(
            "SELECT * FROM assays WHERE discovery_id=? ORDER BY created_at DESC LIMIT 1", (did,)
        )
        if plan_row is None:
            raise ControllerError("discovery has no assay plan")
        plan = json.loads(plan_row["plan_json"])
        if sha256_json({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
            raise ControllerError("assay plan hash mismatch")
        aid = plan_row["id"]
        if plan_row["state"] != "planned":  # re-run: a new assay row with the same plan
            aid = "asy_" + secrets.token_hex(8)
            self.c.db.execute(
                "INSERT INTO assays(id, discovery_id, plan_json, plan_hash, state, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (aid, did, plan_row["plan_json"], plan["plan_hash"], "planned", time.time()),
            )
        self.c.db.execute("UPDATE assays SET state='running' WHERE id=?", (aid,))
        self._event(did, "assay_started", author, {"assay_id": aid, "plan_hash": plan["plan_hash"]})
        self.tasks[did] = asyncio.create_task(self._run(did, aid, plan, author))
        return self.get(did)

    async def _run(self, did: str, aid: str, plan: dict[str, Any], author: str) -> None:
        results: dict[str, dict[str, Any]] = {}
        for check in plan["checks"]:
            try:
                results[check["id"]] = await self.c._in_pool(
                    assays.run_check, check, plan["seeds"], plan["max_alive_fraction"]
                )
            except Exception as exc:  # recorded, never hidden
                results[check["id"]] = {"id": check["id"], "error": f"{type(exc).__name__}: {exc}"}
        verdict = assays.judge(plan, results)
        payload = {"assay_id": aid, "plan_hash": plan["plan_hash"], "results": results,
                   "judgement": verdict, "environment": {
                       "git_commit": self.c.environment.get("git_commit"),
                       "source_sha256": self.c.environment.get("source_digest")}}  # fmt: skip
        row = self._row(did)
        path = f"discoveries/{did}/{aid}.json"
        try:
            receipt = self.c.store.publish_bytes(
                row["source_run_id"], path, canonical_json(payload).encode(), "assay",
                "application/json",
            )  # fmt: skip
            artifact = receipt.logical_path
        except Exception:  # the DB row still holds the full result
            artifact = None
        self.c.db.execute(
            "UPDATE assays SET state='completed', result_json=?, artifact_path=?, finished_at=? "
            "WHERE id=?",
            (canonical_json(payload), artifact, time.time(), aid),
        )
        self._event(did, "assay_completed", author, {"assay_id": aid,
                                                      "status": verdict["status"]})  # fmt: skip
        self._set_status(did, verdict["status"], verdict["evidence_level"], "assay_rule",
                         f"pre-declared rule of plan {plan['plan_hash'][:12]}")  # fmt: skip

    # ------------------------------------------------------------------ views

    def list(self) -> list[dict[str, Any]]:
        rows = self.c.db.all("SELECT * FROM discoveries ORDER BY created_at DESC")
        return [self._summary(dict(r)) for r in rows]

    def _summary(self, r: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": r["id"],
            "title": r["title"],
            "claim_kind": r["claim_kind"],
            "substrate": r["substrate"],
            "status": r["status"],
            "evidence_level": r["evidence_level"],
            "source_run_id": r["source_run_id"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "running": r["id"] in self.tasks and not self.tasks[r["id"]].done(),
        }

    def get(self, did: str) -> dict[str, Any]:
        r = self._row(did)
        out = self._summary(r)
        out.update(
            source=json.loads(r["source_ref_json"]),
            candidate_spec=json.loads(r["candidate_spec_json"]),
            candidate_spec_hash=r["candidate_spec_hash"],
            recorded=json.loads(r["recorded_json"]),
            assays=[
                {"id": a["id"], "state": a["state"], "plan": json.loads(a["plan_json"]),
                 "result": json.loads(a["result_json"]) if a["result_json"] else None,
                 "artifact_path": a["artifact_path"], "created_at": a["created_at"],
                 "finished_at": a["finished_at"]}
                for a in self.c.db.all(
                    "SELECT * FROM assays WHERE discovery_id=? ORDER BY created_at", (did,))
            ],
            events=[
                {"at": e["at"], "kind": e["kind"], "author": e["author"],
                 "payload": json.loads(e["payload_json"])}
                for e in self.c.db.all(
                    "SELECT * FROM discovery_events WHERE discovery_id=? ORDER BY id", (did,))
            ],
        )  # fmt: skip
        return out

    def record(self, did: str) -> dict[str, Any]:
        """Discovery Record export: claim, candidate, provenance, plan, results, history."""
        d = self.get(did)
        return {
            "record_version": 1,
            "discovery": {k: d[k] for k in ("id", "title", "claim_kind", "substrate", "status",
                                            "evidence_level", "created_at", "updated_at")},
            "claim": assays.__doc__,
            "candidate": {"spec": d["candidate_spec"], "spec_sha256": d["candidate_spec_hash"],
                          "recorded": d["recorded"]},
            "provenance": {"source": d["source"],
                           "source_run": self.c.provenance(d["source_run_id"])},
            "assays": d["assays"],
            "history": d["events"],
            "limitations": LIMITATIONS,
        }  # fmt: skip
