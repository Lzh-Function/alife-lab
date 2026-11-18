"""BFF virtual machine: scalar reference oracle and a vectorised NumPy batch interpreter.

Semantics follow CuBFF ``bff.inc.h`` (github.com/paradigms-of-intelligence/cubff, commit
f212e849, Apache-2.0); see ``docs/reference_fidelity/bff.md``. Programs are *data*
interpreted by this VM only — nothing is ever executed on the host.

Tape: 128 bytes (two 64-byte programs). Instructions (ASCII):
  ``<`` ``>`` head0 −/+ 1 · ``{`` ``}`` head1 −/+ 1 · ``-`` ``+`` tape[head0] −/+ 1 (mod 256)
  ``.`` tape[head1] = tape[head0] · ``,`` tape[head0] = tape[head1]
  ``[`` if tape[head0] == 0 jump forward to the matching ``]`` (unmatched: stop)
  ``]`` if tape[head0] != 0 jump back to the matching ``[`` (unmatched: stop)
  any other byte: no-op.
Heads are reduced mod 128 at the start of every step. Variant ``bff_noheads``: heads start
at 0 and pc at 0. Variant ``bff``: head0 = tape[0] mod 128, head1 = tape[1] mod 128, pc = 2.
Execution stops after ``step_limit`` steps or when pc leaves the tape. The returned count is
CuBFF's "ops": steps executed minus no-op steps.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

TAPE = 128
HALF = 64
STEP_LIMIT = 8 * 1024

LT, GT, LB, RB = ord("<"), ord(">"), ord("{"), ord("}")
MINUS, PLUS, DOT, COMMA = ord("-"), ord("+"), ord("."), ord(",")
OPEN, CLOSE = ord("["), ord("]")
INSTRUCTIONS = frozenset({LT, GT, LB, RB, MINUS, PLUS, DOT, COMMA, OPEN, CLOSE})

Variant = Literal["bff_noheads", "bff"]


def evaluate_scalar(
    tape: bytearray, step_limit: int = STEP_LIMIT, variant: Variant = "bff_noheads"
) -> int:
    """Line-by-line port of CuBFF ``Evaluate``/``EvaluateOne`` (reference oracle)."""
    if len(tape) != TAPE:
        raise ValueError("tape must be 128 bytes")
    if variant == "bff":
        head0, head1, pos = tape[0] % TAPE, tape[1] % TAPE, 2
    else:
        head0, head1, pos = TAPE, TAPE, 0
    return _continue_scalar(tape, head0, head1, pos, 0, 0, step_limit)


def _continue_scalar(
    tape: bytearray, head0: int, head1: int, pos: int, i: int, nskip: int, step_limit: int
) -> int:
    """Run from an intermediate VM state (start of step ``i``); returns ops."""
    while i < step_limit:
        head0 &= TAPE - 1
        head1 &= TAPE - 1
        cmd = tape[pos]
        if cmd == LT:
            head0 -= 1
        elif cmd == GT:
            head0 += 1
        elif cmd == LB:
            head1 -= 1
        elif cmd == RB:
            head1 += 1
        elif cmd == PLUS:
            tape[head0] = (tape[head0] + 1) & 0xFF
        elif cmd == MINUS:
            tape[head0] = (tape[head0] - 1) & 0xFF
        elif cmd == DOT:
            tape[head1] = tape[head0]
        elif cmd == COMMA:
            tape[head0] = tape[head1]
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
        else:
            nskip += 1
        if pos < 0:
            i += 1
            break
        pos += 1
        if pos >= TAPE:
            i += 1
            break
        i += 1
    return i - nskip


HANDOFF = 512  # break-even: ~0.4 ms per vectorised step vs ~0.7 µs per scalar tape-step


def evaluate_batch(
    tapes: npt.NDArray[np.uint8],
    step_limit: int = STEP_LIMIT,
    variant: Variant = "bff_noheads",
    handoff: int = HANDOFF,
) -> npt.NDArray[np.int64]:
    """Evaluate many 128-byte tapes in lockstep (in place). Returns ops per tape.

    Bit-for-bit equivalent to :func:`evaluate_scalar` applied to each row. When at most
    ``handoff`` tapes are still running (typically long loops), they are finished by the
    scalar interpreter from their exact intermediate state, which avoids paying the
    vectorised per-step overhead for a handful of tapes.
    """
    if tapes.ndim != 2 or tapes.shape[1] != TAPE or tapes.dtype != np.uint8:
        raise ValueError("tapes must be uint8 of shape (P, 128)")
    p = tapes.shape[0]
    rows = np.arange(p)
    if variant == "bff":
        head0 = tapes[:, 0].astype(np.int64) % TAPE
        head1 = tapes[:, 1].astype(np.int64) % TAPE
        pos = np.full(p, 2, dtype=np.int64)
    else:
        head0 = np.full(p, TAPE, dtype=np.int64)
        head1 = np.full(p, TAPE, dtype=np.int64)
        pos = np.zeros(p, dtype=np.int64)
    steps = np.zeros(p, dtype=np.int64)
    nskip = np.zeros(p, dtype=np.int64)
    active = rows.copy()
    ops = np.zeros(p, dtype=np.int64)
    for _ in range(step_limit):
        if active.size == 0:
            break
        if active.size <= handoff:
            for r in active.tolist():
                row = bytearray(tapes[r].tobytes())
                ops[r] = _continue_scalar(
                    row,
                    int(head0[r]),
                    int(head1[r]),
                    int(pos[r]),
                    int(steps[r]),
                    int(nskip[r]),
                    step_limit,
                )
                tapes[r] = np.frombuffer(bytes(row), dtype=np.uint8)
            done = np.zeros(p, dtype=bool)
            done[active] = True
            finished = ~done
            ops[finished] = steps[finished] - nskip[finished]
            return ops
        head0[active] &= TAPE - 1
        head1[active] &= TAPE - 1
        a = active
        h0, h1, pc = head0[a], head1[a], pos[a]
        cmd = tapes[a, pc]
        head0[a] = h0 - (cmd == LT) + (cmd == GT)
        head1[a] = h1 - (cmd == LB) + (cmd == RB)
        m = cmd == PLUS
        if m.any():
            tapes[a[m], h0[m]] += np.uint8(1)
        m = cmd == MINUS
        if m.any():
            tapes[a[m], h0[m]] -= np.uint8(1)
        m = cmd == DOT
        if m.any():
            tapes[a[m], h1[m]] = tapes[a[m], h0[m]]
        m = cmd == COMMA
        if m.any():
            tapes[a[m], h0[m]] = tapes[a[m], h1[m]]
        m = (cmd == OPEN) & (tapes[a, h0] == 0)
        if m.any():
            pc[m] = _scan_forward(tapes, a[m], pc[m])
        m = (cmd == CLOSE) & (tapes[a, h0] != 0)
        if m.any():
            pc[m] = _scan_backward(tapes, a[m], pc[m])
        nskip[a] += ~np.isin(cmd, _INSTR_ARRAY)
        steps[a] += 1
        stopped = pc < 0
        pc = pc + 1
        stopped |= pc >= TAPE
        pos[a] = pc
        active = a[~stopped]
    return steps - nskip


_INSTR_ARRAY = np.array(sorted(INSTRUCTIONS), dtype=np.uint8)


def _scan_forward(
    tapes: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64], pc: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Matching ``]`` position for ``[`` at pc; TAPE if unmatched (CuBFF semantics)."""
    depth = np.ones(rows.size, dtype=np.int64)
    cur = pc + 1
    result = np.full(rows.size, TAPE, dtype=np.int64)
    open_ = np.ones(rows.size, dtype=bool)
    while open_.any():
        idx = np.flatnonzero(open_)
        inside = cur[idx] < TAPE
        done_out = idx[~inside]
        open_[done_out] = False  # ran off the tape: unmatched -> TAPE
        idx = idx[inside]
        if idx.size == 0:
            break
        c = tapes[rows[idx], cur[idx]]
        depth[idx] -= c == CLOSE
        depth[idx] += c == OPEN
        matched = idx[depth[idx] == 0]
        result[matched] = cur[matched]
        open_[matched] = False
        cur[idx] += 1
    return result


def _scan_backward(
    tapes: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64], pc: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Matching ``[`` position for ``]`` at pc; −1 if unmatched (CuBFF semantics)."""
    depth = np.ones(rows.size, dtype=np.int64)
    cur = pc - 1
    result = np.full(rows.size, -1, dtype=np.int64)
    open_ = np.ones(rows.size, dtype=bool)
    while open_.any():
        idx = np.flatnonzero(open_)
        inside = cur[idx] >= 0
        open_[idx[~inside]] = False
        idx = idx[inside]
        if idx.size == 0:
            break
        c = tapes[rows[idx], cur[idx]]
        depth[idx] += c == CLOSE
        depth[idx] -= c == OPEN
        matched = idx[depth[idx] == 0]
        result[matched] = cur[matched]
        open_[matched] = False
        cur[idx] -= 1
    return result


def disassemble(program: bytes) -> str:
    """Printable view: instructions as themselves, NUL as '0', other bytes as '·'."""
    return "".join(chr(b) if b in INSTRUCTIONS else ("0" if b == 0 else "·") for b in program)
