"""BFF replicator assay: does a program make copies of itself, compared with controls?

A passive, pre-registered measurement on *copies* of a program (it never touches a run):

* **pair assay** — evaluate ``P + R`` and ``R + P`` for random partners ``R`` (and an all-NUL
  partner); report how often the partner half becomes an exact (or near) copy of ``P`` and how
  often ``P`` survives; plus the self pair ``P + P``.
* **soup assay** — a random background soup (CuBFF initialisation) with ``k`` injected copies of
  ``P`` runs ``E`` CuBFF epochs; exact and near (Hamming ≤ ``near_distance``) copies are counted
  every epoch. Controls with the same background: a byte-scrambled ``P`` (same composition) and
  no injection. Replicates use different assay seeds.

Verdict rule (fixed before the assay is run, stored with the result): ``replicates_in_assay``
iff, in a strict majority of replicates, the final exact copy count of ``P`` is at least
``growth_factor × k`` while the scrambled control's is not. ``inconclusive`` if the scrambled
program equals ``P`` or the scrambled control also grows. This is evidence for copying in this
assay only — not a proof of open-ended replication.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alife_lab.substrates.bff.soup import init_soup, run_epoch
from alife_lab.substrates.bff.vm import HALF, STEP_LIMIT, evaluate_batch

U8 = npt.NDArray[np.uint8]
MAX_PAIR_EVALUATIONS = 2_000_000  # budget for the soup assay (pairs x epochs x conditions)


class BFFAssayParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=0, ge=0, lt=2**63)
    variant: Literal["bff_noheads", "bff"] = "bff_noheads"
    step_limit: int = Field(default=STEP_LIMIT, ge=0, le=STEP_LIMIT)
    partners: int = Field(default=256, ge=1, le=4096)
    soup_size: int = Field(default=1024, ge=64, le=4096, multiple_of=2)
    copies: int = Field(default=16, ge=1, le=2048)
    epochs: int = Field(default=64, ge=1, le=512)
    mutation_prob: int = Field(default=0, ge=0, le=1 << 30)
    replicates: int = Field(default=3, ge=1, le=7)
    near_distance: int = Field(default=4, ge=0, le=32)
    growth_factor: float = Field(default=4.0, gt=1.0, le=1000.0)

    @model_validator(mode="after")
    def _check(self) -> BFFAssayParams:
        if self.copies > self.soup_size // 2:
            raise ValueError("copies must be at most half of soup_size")
        cost = (self.soup_size // 2) * self.epochs * self.replicates * 3
        if cost > MAX_PAIR_EVALUATIONS:
            raise ValueError(
                f"assay too large: {cost} pair evaluations > {MAX_PAIR_EVALUATIONS} "
                "(reduce soup_size, epochs or replicates)"
            )
        return self


def _evaluate(tapes: U8, step_limit: int, variant: str, backend: str) -> npt.NDArray[np.int64]:
    if backend == "numba":
        from alife_lab.substrates.bff.vm_numba import evaluate_batch_numba

        return evaluate_batch_numba(tapes, step_limit, variant)
    return evaluate_batch(tapes, step_limit, variant)  # type: ignore[arg-type]


def _backend() -> str:
    import importlib.util

    return "numba" if importlib.util.find_spec("numba") is not None else "numpy"


def pair_assay(program: bytes, p: BFFAssayParams, backend: str) -> dict[str, Any]:
    prog = np.frombuffer(program, dtype=np.uint8)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, 1])))
    partners = rng.integers(0, 256, size=(p.partners, HALF), dtype=np.uint8)
    partners = np.vstack([np.zeros((1, HALF), dtype=np.uint8), partners])  # row 0: all-NUL

    def run(first: bool) -> dict[str, Any]:
        pp = np.broadcast_to(prog, partners.shape)
        tapes = np.ascontiguousarray(np.hstack([pp, partners] if first else [partners, pp]))
        ops = _evaluate(tapes, p.step_limit, p.variant, backend)
        mine = tapes[:, :HALF] if first else tapes[:, HALF:]
        other = tapes[:, HALF:] if first else tapes[:, :HALF]
        copied = (other == prog).all(axis=1)
        intact = (mine == prog).all(axis=1)
        dist = (other != prog).sum(axis=1)
        return {
            "nul_partner": {
                "partner_became_copy": bool(copied[0]),
                "program_intact": bool(intact[0]),
            },
            "random_partners": {
                "n": int(p.partners),
                "copy_rate": float(copied[1:].mean()),
                "intact_rate": float(intact[1:].mean()),
                "copy_and_intact_rate": float((copied & intact)[1:].mean()),
                "near_copy_rate": float((dist <= p.near_distance)[1:].mean()),
                "mean_partner_hamming_to_program": float(dist[1:].mean()),
                "mean_ops": float(ops[1:].mean()),
            },
        }  # fmt: skip

    self_tape = np.concatenate([prog, prog])[None, :].copy()
    self_ops = _evaluate(self_tape, p.step_limit, p.variant, backend)
    self_pair = {
        "first_half_intact": bool((self_tape[0, :HALF] == prog).all()),
        "second_half_intact": bool((self_tape[0, HALF:] == prog).all()),
        "hamming_first": int((self_tape[0, :HALF] != prog).sum()),
        "hamming_second": int((self_tape[0, HALF:] != prog).sum()),
        "ops": int(self_ops[0]),
    }
    return {"program_first": run(True), "program_second": run(False), "self_pair": self_pair}


def _hamming_counts(soup: U8, target: U8, near: int) -> tuple[int, int]:
    dist = (soup != target).sum(axis=1)
    return int((dist == 0).sum()), int((dist <= near).sum())


def soup_assay(program: bytes, scrambled: bytes, p: BFFAssayParams, backend: str) -> dict[str, Any]:
    prog = np.frombuffer(program, dtype=np.uint8)
    scr = np.frombuffer(scrambled, dtype=np.uint8)
    replicates = []
    for r in range(p.replicates):
        assay_seed = (p.seed * 1_000_003 + r * 7919 + 17) % (2**63)
        background = init_soup(assay_seed, p.soup_size)
        conditions: dict[str, dict[str, Any]] = {}
        for name, injected, target in (
            ("program", prog, prog),
            ("scrambled", scr, scr),
            ("background", None, prog),
        ):
            soup = background.copy()
            if injected is not None:
                soup[: p.copies] = injected
            exact, near = [], []
            e0, n0 = _hamming_counts(soup, target, p.near_distance)
            exact.append(e0)
            near.append(n0)
            for epoch in range(p.epochs):
                run_epoch(
                    soup, assay_seed, epoch, p.mutation_prob, p.step_limit, p.variant, backend
                )
                e, n = _hamming_counts(soup, target, p.near_distance)
                exact.append(e)
                near.append(n)
            conditions[name] = {"exact": exact, "near": near, "final_exact": exact[-1]}
        replicates.append({"assay_seed": assay_seed, "conditions": conditions})
    return {"replicates": replicates}


def verdict(
    program: bytes, scrambled: bytes, soup: dict[str, Any], p: BFFAssayParams
) -> dict[str, Any]:
    threshold = p.growth_factor * p.copies
    grew = [r["conditions"]["program"]["final_exact"] >= threshold for r in soup["replicates"]]
    ctrl = [r["conditions"]["scrambled"]["final_exact"] >= threshold for r in soup["replicates"]]
    majority = len(grew) // 2 + 1
    if program == scrambled:
        outcome, reason = "inconclusive", "scrambled control equals the program"
    elif sum(ctrl) >= majority:
        outcome, reason = "inconclusive", "the scrambled control also grew"
    elif sum(g and not c for g, c in zip(grew, ctrl, strict=True)) >= majority:
        outcome, reason = "replicates_in_assay", "program grew, scrambled control did not"
    else:
        outcome, reason = "no_evidence", "program did not grow past the threshold"
    return {
        "outcome": outcome,
        "reason": reason,
        "rule": (
            f"final exact copies >= {p.growth_factor} x {p.copies} = {threshold:g} in a strict "
            f"majority ({majority}/{len(grew)}) of replicates, scrambled control below it"
        ),
        "program_grew": grew,
        "scrambled_grew": ctrl,
    }


def run_assay(program_hex: str, params: dict[str, Any]) -> dict[str, Any]:
    """Entry point for the controller's worker pool (picklable arguments only)."""
    p = BFFAssayParams.model_validate(params)
    program = bytes.fromhex(program_hex)
    if len(program) != HALF:
        raise ValueError("program must be 64 bytes")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, 2])))
    scrambled = bytes(np.frombuffer(program, dtype=np.uint8)[rng.permutation(HALF)])
    backend = _backend()
    started = time.perf_counter()
    pairs = pair_assay(program, p, backend)
    soup = soup_assay(program, scrambled, p, backend)
    return {
        "assay": "bff_replicator",
        "assay_version": 1,
        "program_hex": program.hex(),
        "scrambled_hex": scrambled.hex(),
        "params": p.model_dump(),
        "backend": backend,
        "pair_assay": pairs,
        "soup_assay": soup,
        "verdict": verdict(program, scrambled, soup, p),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
