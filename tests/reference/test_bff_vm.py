"""BFF VM: instruction semantics (CuBFF bff.inc.h) and batch == scalar oracle."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alife_lab.substrates.bff.vm import (
    TAPE,
    disassemble,
    evaluate_batch,
    evaluate_scalar,
)


def tape_of(prog: str, fill: int = 1) -> bytearray:
    """Program text at position 0, rest filled with a no-op byte (1)."""
    t = bytearray([fill] * TAPE)
    t[: len(prog)] = prog.encode("latin-1")
    return t


def test_heads_start_at_zero_and_wrap() -> None:
    t = tape_of("<+")  # head0: 0 -> -1 -> masked to 127, then +1 at 127
    ops = evaluate_scalar(t)
    assert t[127] == 2 and ops == 2


def test_increment_decrement_wrap_mod_256() -> None:
    t = tape_of(">-")  # head0 -> 1 ; tape[1] = '-'(45) - 1
    evaluate_scalar(t)
    assert t[1] == ord("-") - 1
    t = tape_of(">>" + "-", fill=0)  # tape[2]='-' -> 44 (','), but head0 at 2: decrement '-'
    evaluate_scalar(t)
    assert t[2] == ord("-") - 1


def test_copy_instructions() -> None:
    # head1 -> 5, then copy tape[head0=0] ('}') to tape[5]
    t = tape_of("}}}}}.")
    evaluate_scalar(t)
    assert t[5] == ord("}")
    # ',' copies tape[head1] into tape[head0]
    t = tape_of(">}},")  # head0=1, head1=2 -> tape[1] = tape[2] = '}'
    evaluate_scalar(t)
    assert t[1] == ord("}")


def test_loop_skips_when_zero_and_unmatched_stops() -> None:
    # head0 at 127 (NUL) -> '[' jumps to matching ']' ; '+' inside is skipped
    t = tape_of("<[+]>+", fill=0)
    evaluate_scalar(t)
    assert t[127] == 0  # loop body skipped
    assert t[0] == ord("<") + 1  # after '>' head0 = 0 -> '+' increments tape[0]
    t = tape_of("<[++", fill=0)  # unmatched '[' with zero -> stop immediately
    ops = evaluate_scalar(t)
    assert ops == 2 and t[0] == ord("<")


def test_backward_jump_loops_until_zero() -> None:
    # tape[127] counts down from 3 via a loop at the start of the tape.
    t = tape_of("<[-]", fill=0)
    t[127] = 3
    ops = evaluate_scalar(t)
    assert t[127] == 0
    assert ops == 1 + 1 + 3 * 2  # '<', '[' and three '-',']' pairs


def test_unmatched_close_stops() -> None:
    t = tape_of("+]", fill=0)  # tape[0] becomes '+'+1 != 0, ']' has no '[' -> stop
    assert evaluate_scalar(t) == 2


def test_step_limit() -> None:
    t = tape_of("+[]", fill=0)  # tape[0] = '+'+1 != 0: '[' falls through, ']' jumps back forever
    assert evaluate_scalar(t, step_limit=100) == 100


def test_bff_variant_reads_heads_from_tape() -> None:
    t = tape_of("\x05\x07.")  # head0 = 5, head1 = 7, pc = 2: tape[7] = tape[5]
    t[5] = 42
    evaluate_scalar(t, variant="bff")
    assert t[7] == 42
    t = tape_of("\x85\x07.")  # 0x85 % 128 = 5
    t[5] = 9
    evaluate_scalar(t, variant="bff")
    assert t[7] == 9


def test_disassemble() -> None:
    assert disassemble(b"[\x00a>") == "[0·>"


alphabet = st.sampled_from([*b"<>{}+-.,[]", 0, 0, 1, 255, 91, 93])


@settings(max_examples=300, deadline=None)
@given(
    data=st.lists(alphabet, min_size=TAPE, max_size=TAPE),
    limit=st.sampled_from([1, 7, 64, 500, 8192]),
    variant=st.sampled_from(["bff_noheads", "bff"]),
)
def test_batch_equals_scalar_oracle(data: list[int], limit: int, variant: str) -> None:
    scalar = bytearray(data)
    ops_s = evaluate_scalar(scalar, limit, variant)  # type: ignore[arg-type]
    batch = np.array([data, data[::-1]], dtype=np.uint8)
    # handoff=0: exercise the pure vectorised path (the default hands 2 tapes to scalar).
    ops_b = evaluate_batch(batch, limit, variant, handoff=0)  # type: ignore[arg-type]
    assert bytes(batch[0]) == bytes(scalar)
    assert int(ops_b[0]) == ops_s
    rev = bytearray(data[::-1])
    assert int(ops_b[1]) == evaluate_scalar(rev, limit, variant)  # type: ignore[arg-type]
    assert bytes(batch[1]) == bytes(rev)


def test_batch_equals_scalar_on_uniform_random_bytes() -> None:
    rng = np.random.default_rng(0)
    tapes = rng.integers(0, 256, size=(400, TAPE), dtype=np.uint8)
    ref = [bytearray(t.tobytes()) for t in tapes]
    ops_ref = [evaluate_scalar(t) for t in ref]
    ops = evaluate_batch(tapes.copy(), handoff=0)
    assert ops.tolist() == ops_ref
    hybrid = tapes.copy()
    assert evaluate_batch(hybrid, handoff=64).tolist() == ops_ref  # vectorised -> scalar
    tapes = hybrid
    assert all(bytes(a) == bytes(b) for a, b in zip(tapes, ref, strict=True))


def test_batch_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        evaluate_batch(np.zeros((2, 64), dtype=np.uint8))
