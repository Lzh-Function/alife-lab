"""BFF debugger trace: equal to the scalar oracle, self-consistent write log, halt reasons, and
the next-epoch pair reconstruction equal to what run_epoch computes."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alife_lab.substrates.bff.soup import init_soup, run_epoch
from alife_lab.substrates.bff.trace import next_epoch_pair, trace_scalar
from alife_lab.substrates.bff.vm import HALF, evaluate_scalar

OPS = b"<>{}+-.,[]"
tape_bytes = st.lists(
    st.one_of(st.sampled_from(list(OPS)), st.integers(0, 255)), min_size=128, max_size=128
).map(bytes)


@settings(max_examples=300, deadline=None)
@given(tape=tape_bytes, limit=st.integers(0, 2000), variant=st.sampled_from(["bff_noheads", "bff"]))
def test_trace_equals_scalar_oracle(tape: bytes, limit: int, variant: str) -> None:
    ref = bytearray(tape)
    ops = evaluate_scalar(ref, limit, variant)  # type: ignore[arg-type]
    tr = trace_scalar(tape, limit, variant)  # type: ignore[arg-type]
    assert tr["ops"] == ops
    assert bytes.fromhex(tr["final_tape"]) == bytes(ref)
    assert len(tr["pc"]) == tr["steps"] <= limit
    # Replaying the write log from the initial tape reproduces the final tape.
    cur = bytearray(tape)
    for addr, old, new in zip(tr["write_addr"], tr["write_old"], tr["write_new"], strict=True):
        if addr >= 0:
            assert cur[addr] == old
            cur[addr] = new
    assert bytes(cur) == bytes(ref)
    # Control flow is consistent: each step starts where the previous one said it would.
    assert all(n == p for n, p in zip(tr["next_pc"][:-1], tr["pc"][1:], strict=True))


@pytest.mark.parametrize(
    ("program", "halt"),
    [
        (b"<[", "unmatched_open"),  # head0 -> 127 (NUL), no matching ']'
        (b"+]", "unmatched_close"),  # tape[0] != 0, no matching '['
        (b"+", "end_of_tape"),  # then 127 no-ops
        (b"[]", "step_limit"),  # tape[0] = '[' != 0 forever: infinite loop
    ],
)
def test_halt_reasons(program: bytes, halt: str) -> None:
    tape = program + bytes(128 - len(program))
    assert trace_scalar(tape, 1000)["halt"] == halt


def test_next_epoch_pair_trace_reproduces_run_epoch() -> None:
    n, seed, epoch, mut = 64, 5, 7, 1 << 24  # high mutation rate: several mutated bytes
    soup = init_soup(seed, n)
    for e in range(epoch):
        run_epoch(soup, seed, e, mut)
    before = soup.copy()
    after = soup.copy()
    run_epoch(after, seed, epoch, mut)
    any_mutated = False
    for index in (0, 13, 63):
        pair = next_epoch_pair(before, seed, epoch, mut, index)
        tr = trace_scalar(pair["tape"])
        final = bytes.fromhex(tr["final_tape"])
        assert final[:HALF] == after[pair["first"]].tobytes()
        assert final[HALF:] == after[pair["second"]].tobytes()
        assert index in (pair["first"], pair["second"])
        any_mutated |= bool(pair["mutated_positions"])
    assert any_mutated


def test_hand_written_copier_copies_a_into_b() -> None:
    # bff variant: head0 = 0x80 % 128 = 0, head1 = 0xC0 % 128 = 64, pc = 2; "[.>}]" copies
    # tape[head0] -> tape[head1] while the byte under head0 is non-zero (7 bytes).
    a = bytes.fromhex("80c05b2e3e7d5d") + bytes(57)
    tr = trace_scalar(a + bytes(64), 8192, "bff")
    final = bytes.fromhex(tr["final_tape"])
    assert final[:HALF] == a and final[HALF:] == a and tr["halt"] == "end_of_tape"
