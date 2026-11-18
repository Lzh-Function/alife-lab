"""BFF substrate adapter (clock unit: epoch).

Reference mode reproduces CuBFF's soup procedure bit-for-bit (VM, SplitMix64 RNG, shuffle,
mutation); see ``soup.py`` and ``docs/reference_fidelity/bff.md``. Metrics are passive and
never feed back into the soup: there is no fitness, no selection and no reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import brotli
import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import BFFExperiment, ExperimentSpec
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.bff.soup import init_soup, run_epoch
from alife_lab.substrates.bff.vm import HALF, TAPE, disassemble, evaluate_scalar

SERIALIZE_VERSION = 1
MAX_DISPLAY_PROGRAMS = 4096
MAX_INTERVENTION_SLOTS = 4096


def _bff(spec: ExperimentSpec) -> BFFExperiment:
    exp = spec.experiment
    if not isinstance(exp, BFFExperiment):
        raise TypeError("bff adapter received a non-bff spec")
    return exp


@dataclass
class BFFState:
    soup: npt.NDArray[np.uint8]  # (N, 64)
    tick: int  # epochs completed == index of the next epoch
    last: dict[str, Any] = field(default_factory=dict)


def display_columns(num_programs: int) -> int:
    """Tile grid width used by the GUI: ceil(sqrt(shown programs))."""
    shown = min(num_programs, MAX_DISPLAY_PROGRAMS)
    return max(1, math.ceil(math.sqrt(shown)))


def soup_entropy_metrics(soup: npt.NDArray[np.uint8]) -> dict[str, float]:
    """CuBFF statistics: h0 (bits/byte), brotli q2/lgwin24 bits/byte, their difference."""
    data = soup.tobytes()
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    frac = counts[counts > 0] / len(data)
    h0 = float(-(frac * np.log2(frac)).sum())
    size = len(brotli.compress(data, mode=brotli.MODE_GENERIC, quality=2, lgwin=24))
    bpb = size * 8.0 / len(data)
    return {
        "h0_bits_per_byte": h0,
        "brotli_bits_per_byte": bpb,
        "higher_order_entropy": h0 - bpb,
    }


_M = MetricDefinition
_METRICS = (
    _M(
        "h0_bits_per_byte",
        "metric.h0_bits_per_byte",
        "bits/byte",
        "epoch",
        "Shannon entropy of the byte histogram of the whole soup (CuBFF h0).",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "brotli_bits_per_byte",
        "metric.brotli_bits_per_byte",
        "bits/byte",
        "epoch",
        "Brotli (quality 2, lgwin 24, generic mode) compressed size of the soup in bits per "
        "byte — a computable stand-in for normalised Kolmogorov complexity. Library version "
        "is recorded in the environment; sizes may differ slightly across brotli versions.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "higher_order_entropy",
        "metric.higher_order_entropy",
        "bits/byte",
        "epoch",
        "h0 minus brotli bits/byte (CuBFF 'higher entropy'). Rises when the soup becomes "
        "repetitive at the program level. A complexity proxy, not a replication detector.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "ops_per_pair",
        "metric.ops_per_pair",
        "instructions",
        "epoch",
        "Mean executed non-no-op instructions per pair in the last epoch (CuBFF ops).",
        "mean over pairs",
        "every epoch",
        1,
        "null at epoch 0",
    ),
    _M(
        "unique_programs",
        "metric.unique_programs",
        "programs",
        "epoch",
        "Number of distinct 64-byte programs in the soup.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "top_program_fraction",
        "metric.top_program_fraction",
        "fraction",
        "epoch",
        "Share of the soup occupied by the most frequent exact program. Exact copies are a "
        "hint, not evidence, of replication.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "mutated_bytes",
        "metric.mutated_bytes",
        "bytes",
        "epoch",
        "Bytes replaced by background mutation in the last epoch.",
        "sum",
        "every epoch",
        1,
        "null at epoch 0",
    ),
)

BFF_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.bff",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="bff",
    display_name_key="substrate.bff",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="program_soup",
        clock_unit="epoch",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("inject_program", "zero_programs"),
    ),
    # Order = preference for backend "auto"; both are bit-identical (tested).
    backends=("numba", "numpy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Agüera y Arcas et al. (2024) Computational Life; CuBFF bff / "
        "bff_noheads (Apache-2.0).",
        "paper_urls": ["https://arxiv.org/abs/2406.19108"],
        "implementation": "https://github.com/paradigms-of-intelligence/cubff",
        "implementation_commit": "f212e849027c98fcf4b242eccfb5fed435223e23",
    },
)


def _parse_program(text: Any, encoding: str) -> bytes:
    if not isinstance(text, str):
        raise ValueError("program must be a string")
    if encoding == "hex":
        try:
            raw = bytes.fromhex(text)
        except ValueError as exc:
            raise ValueError("invalid hex program") from exc
    elif encoding == "text":
        try:
            raw = text.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError("text programs must be latin-1") from exc
    else:
        raise ValueError("encoding must be 'text' or 'hex'")
    if len(raw) > HALF:
        raise ValueError(f"program longer than {HALF} bytes")
    return raw + bytes(HALF - len(raw))  # pad with NUL


def _slots(payload: dict[str, Any], n: int) -> list[int]:
    raw = payload.get("indices")
    if not isinstance(raw, list) or not raw or len(raw) > MAX_INTERVENTION_SLOTS:
        raise ValueError(f"indices must be a non-empty list of at most {MAX_INTERVENTION_SLOTS}")
    out = sorted({int(i) for i in raw})
    if out[0] < 0 or out[-1] >= n:
        raise ValueError("program index out of range")
    return out


class BFFAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        """Execution-only choice; never changes results (backends are bit-identical)."""
        if backend not in BFF_MANIFEST.backends:
            raise ValueError(f"unsupported bff backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return BFF_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> BFFState:
        exp = _bff(spec)
        soup = init_soup(exp.seed, exp.soup.num_programs, zero=exp.soup.init == "zero")
        return BFFState(soup=soup, tick=0)

    def step(self, state: BFFState, spec: ExperimentSpec) -> BFFState:
        exp = _bff(spec)
        state.last = run_epoch(
            state.soup,
            exp.seed,
            state.tick,
            exp.dynamics.mutation_prob,
            exp.dynamics.step_limit,
            exp.dynamics.variant,
            backend=self.backend,
        )
        state.tick += 1
        return state

    def metrics(self, state: BFFState, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_bff(spec).observations.metrics)
        out: dict[str, float] = {}
        if wanted & {"h0_bits_per_byte", "brotli_bits_per_byte", "higher_order_entropy"}:
            out.update(soup_entropy_metrics(state.soup))
        if wanted & {"unique_programs", "top_program_fraction"}:
            # Rows as opaque 64-byte values: same counts as unique(axis=0), ~3x faster.
            rows = np.ascontiguousarray(state.soup).view(np.dtype((np.void, HALF))).ravel()
            _, counts = np.unique(rows, return_counts=True)
            out["unique_programs"] = float(counts.size)
            out["top_program_fraction"] = float(counts.max() / state.soup.shape[0])
        out["ops_per_pair"] = float(state.last["ops_per_pair"]) if state.last else math.nan
        out["mutated_bytes"] = float(state.last["mutated_bytes"]) if state.last else math.nan
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: BFFState) -> tuple[dict[str, Any], bytes]:
        n = state.soup.shape[0]
        shown = min(n, MAX_DISPLAY_PROGRAMS)
        return {
            "codec": "bff_soup",
            "programs": shown,
            "total_programs": n,
            "tape": HALF,
            "columns": display_columns(n),
        }, state.soup[:shown].tobytes()

    def apply_intervention(
        self, state: BFFState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[BFFState, dict[str, Any]]:
        n = state.soup.shape[0]
        slots = _slots(payload, n)
        if kind == "inject_program":
            program = _parse_program(payload.get("program"), str(payload.get("encoding", "text")))
            state.soup[slots] = np.frombuffer(program, dtype=np.uint8)
            return state, {"slots": len(slots), "program_hex": program.hex()}
        if kind == "zero_programs":
            state.soup[slots] = 0
            return state, {"slots": len(slots)}
        raise ValueError(f"unsupported intervention kind {kind!r}")

    def inspect(
        self, state: BFFState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        exp = _bff(spec)
        n = state.soup.shape[0]
        cols = display_columns(n)
        index = int(query["y"]) * cols + int(query["x"])
        if not 0 <= index < min(n, MAX_DISPLAY_PROGRAMS):
            raise ValueError("no program at this position")
        program = state.soup[index].tobytes()
        copies = int((state.soup == state.soup[index]).all(axis=1).sum())
        tape = bytearray(program + program)
        ops = evaluate_scalar(tape, exp.dynamics.step_limit, exp.dynamics.variant)
        return {
            "substrate": "bff",
            "tick": state.tick,
            "index": index,
            "program_hex": program.hex(),
            "disassembly": disassemble(program),
            "instruction_fraction": float(
                sum(1 for c in disassemble(program) if c not in "·0") / HALF
            ),
            "exact_copies_in_soup": copies,
            "self_pair_test": {
                "description": "exec(A+A) once with the run's step limit; whether each half "
                "equals A afterwards. Not a replication assay.",
                "ops": ops,
                "first_half_unchanged": bytes(tape[:HALF]) == program,
                "second_half_equals_program": bytes(tape[HALF:]) == program,
            },
        }

    def state_distance(self, a: BFFState, b: BFFState) -> dict[str, Any]:
        if a.soup.shape != b.soup.shape:
            return {"metric": "differing_bytes", "value": None, "exact_equal": False}
        diff = int(np.count_nonzero(a.soup != b.soup))
        return {"metric": "differing_bytes", "value": diff, "exact_equal": diff == 0}

    def serialize_state(self, state: BFFState) -> dict[str, Any]:
        return {
            "arrays": {"soup": state.soup},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "clock_unit": "epoch",
                "last_epoch": state.last,
                "num_programs": int(state.soup.shape[0]),
                "tape_size": TAPE // 2,
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> BFFState:
        exp = _bff(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible bff checkpoint serialize_version")
        soup = np.asarray(payload["arrays"]["soup"], dtype=np.uint8).copy()
        if soup.shape != (exp.soup.num_programs, HALF):
            raise ValueError("bff checkpoint soup shape does not match the spec")
        return BFFState(soup=soup, tick=int(meta["tick"]), last=dict(meta.get("last_epoch") or {}))
