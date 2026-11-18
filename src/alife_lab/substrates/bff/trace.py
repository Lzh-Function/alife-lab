"""Step-by-step trace of one BFF pair evaluation (read-only debugger).

``trace_scalar`` re-runs the scalar oracle loop of ``vm.evaluate_scalar`` and records every
step in columnar form; tests check that its final tape and op count equal the oracle's. The
trace is a view of the VM, never a separate semantics.

``next_epoch_pair`` reconstructs exactly the pair tape that program ``index`` will be part of in
the next epoch (CuBFF shuffle + background mutation of that pair), so the trace of that tape
is what ``run_epoch`` will compute for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.substrates.bff.soup import mutate_pairs, shuffle_indices
from alife_lab.substrates.bff.vm import (
    CLOSE,
    COMMA,
    DOT,
    GT,
    HALF,
    LB,
    LT,
    MINUS,
    OPEN,
    PLUS,
    RB,
    STEP_LIMIT,
    TAPE,
    Variant,
)

MAX_TRACE_STEPS = STEP_LIMIT


def trace_scalar(
    tape_in: bytes, step_limit: int = STEP_LIMIT, variant: Variant = "bff_noheads"
) -> dict[str, Any]:
    """Evaluate ``tape_in`` like ``evaluate_scalar`` and return the full step record."""
    if len(tape_in) != TAPE:
        raise ValueError("tape must be 128 bytes")
    if not 0 <= step_limit <= MAX_TRACE_STEPS:
        raise ValueError(f"step_limit must be within 0..{MAX_TRACE_STEPS}")
    tape = bytearray(tape_in)
    if variant == "bff":
        head0, head1, pos = tape[0] % TAPE, tape[1] % TAPE, 2
    else:
        head0, head1, pos = TAPE, TAPE, 0
    rec: dict[str, list[int]] = {
        k: []
        for k in ("pc", "cmd", "head0", "head1", "write_addr", "write_old", "write_new", "next_pc")
    }
    i = nskip = 0
    halt = "step_limit"
    while i < step_limit:
        head0 &= TAPE - 1
        head1 &= TAPE - 1
        h0, h1, pc = head0, head1, pos
        cmd = tape[pos]
        addr = -1
        old = -1
        if cmd == LT:
            head0 -= 1
        elif cmd == GT:
            head0 += 1
        elif cmd == LB:
            head1 -= 1
        elif cmd == RB:
            head1 += 1
        elif cmd in (PLUS, MINUS, COMMA):
            addr, old = head0, tape[head0]
            if cmd == PLUS:
                tape[head0] = (old + 1) & 0xFF
            elif cmd == MINUS:
                tape[head0] = (old - 1) & 0xFF
            else:
                tape[head0] = tape[head1]
        elif cmd == DOT:
            addr, old = head1, tape[head1]
            tape[head1] = tape[head0]
        elif cmd == OPEN:
            if tape[head0] == 0:
                depth = 1
                pos += 1
                while pos < TAPE and depth > 0:
                    if tape[pos] == CLOSE:
                        depth -= 1
                    if tape[pos] == OPEN:
                        depth += 1
                    pos += 1
                pos -= 1
                if depth != 0:
                    pos = TAPE
                    halt = "unmatched_open"
        elif cmd == CLOSE:
            if tape[head0] != 0:
                depth = 1
                pos -= 1
                while pos >= 0 and depth > 0:
                    if tape[pos] == CLOSE:
                        depth += 1
                    if tape[pos] == OPEN:
                        depth -= 1
                    pos -= 1
                pos += 1
                if depth != 0:
                    pos = -1
                    halt = "unmatched_close"
        else:
            nskip += 1
        rec["pc"].append(pc)
        rec["cmd"].append(cmd)
        rec["head0"].append(h0)  # heads at the start of the step (after the mod-128 reduction)
        rec["head1"].append(h1)
        rec["write_addr"].append(addr)
        rec["write_old"].append(old)
        rec["write_new"].append(tape[addr] if addr >= 0 else -1)
        i += 1
        if pos < 0:
            rec["next_pc"].append(-1)
            break
        pos += 1
        rec["next_pc"].append(pos)
        if pos >= TAPE:
            if halt == "step_limit":
                halt = "end_of_tape"
            break
    return {
        "variant": variant,
        "step_limit": step_limit,
        "initial_tape": bytes(tape_in).hex(),
        "final_tape": bytes(tape).hex(),
        "steps": i,
        "ops": i - nskip,
        "halt": halt,
        **rec,
    }


def next_epoch_pair(
    soup: npt.NDArray[np.uint8], params_seed: int, epoch: int, mutation_prob: int, index: int
) -> dict[str, Any]:
    """The tape ``run_epoch(epoch)`` will evaluate for the pair containing ``index``."""
    n = soup.shape[0]
    if not 0 <= index < n:
        raise ValueError("program index out of range")
    s = shuffle_indices(params_seed, epoch, n)
    where = int(np.flatnonzero(s == index)[0])
    k = where // 2
    first, second = int(s[2 * k]), int(s[2 * k + 1])
    tape = np.concatenate([soup[first], soup[second]])[None, :].copy()
    before = tape.copy()
    mutate_pairs(tape, params_seed, epoch, n, mutation_prob, pair_offset=k)
    mutated = np.flatnonzero(tape[0] != before[0]).tolist()
    return {
        "epoch": epoch,
        "pair": k,
        "first": first,
        "second": second,
        "side": "first" if first == index else "second",
        "tape": tape[0].tobytes(),
        "mutated_positions": [int(p) for p in mutated],
    }


def trace_summary(trace: dict[str, Any], program_a: bytes, program_b: bytes) -> dict[str, Any]:
    final = bytes.fromhex(trace["final_tape"])
    a2, b2 = final[:HALF], final[HALF:]
    return {
        "first_half_unchanged": a2 == program_a,
        "second_half_unchanged": b2 == program_b,
        "second_half_equals_first_input": b2 == program_a,
        "first_half_equals_second_input": a2 == program_b,
        "bytes_changed": sum(
            1 for x, y in zip(final, program_a + program_b, strict=True) if x != y
        ),
    }
