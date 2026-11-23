import { describe, expect, it } from "vitest";
import { bitAt, circleCells, popcount, unpackFrame, unpackFrames } from "./api";
import { DICTIONARIES, MESSAGE_KEYS } from "../i18n";

describe("frame decoding", () => {
  it("unpacks header and numpy.packbits payload (big-endian bit order)", () => {
    const header = new TextEncoder().encode(JSON.stringify({ tick: 3, width: 4, height: 2, codec: "bitpack1" }));
    const buf = new Uint8Array(4 + header.length + 1);
    new DataView(buf.buffer).setUint32(0, header.length, false);
    buf.set(header, 4);
    buf[4 + header.length] = 0b10100001;
    const frame = unpackFrame(buf.buffer);
    expect(frame.header.tick).toBe(3);
    expect([0, 1, 2, 3, 4, 5, 6, 7].map((i) => bitAt(frame.bits, i))).toEqual([1, 0, 1, 0, 0, 0, 0, 1]);
  });

  it("handles float32 payloads safely even when JSON header length is not a multiple of 4", () => {
    const header = new TextEncoder().encode(JSON.stringify({ tick: 7, codec: "fly_swarm_v1", num_flies: 1, x: "abc" }));
    expect(header.length % 4).not.toBe(0);
    const payload = new Float32Array([1.25, -2.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.75]);
    const payloadBytes = new Uint8Array(payload.buffer);
    const buf = new Uint8Array(4 + header.length + payloadBytes.length);
    new DataView(buf.buffer).setUint32(0, header.length, false);
    buf.set(header, 4);
    buf.set(payloadBytes, 4 + header.length);
    const frame = unpackFrame(buf.buffer);
    const aligned = new Uint8Array(frame.bits.byteLength);
    aligned.set(frame.bits);
    const decoded = new Float32Array(aligned.buffer);
    expect(decoded[0]).toBeCloseTo(1.25);
    expect(decoded[1]).toBeCloseTo(-2.5);
    expect(decoded[7]).toBeCloseTo(0.75);
  });
});

describe("circleCells", () => {
  it("matches the server rule (radius 1 => plus shape, clipped at edges)", () => {
    expect(circleCells(4, 4, 1, 8, 8)).toHaveLength(5);
    expect(circleCells(0, 0, 1, 8, 8)).toHaveLength(3);
    expect(circleCells(2, 2, 0, 8, 8)).toEqual([[2, 2]]);
  });
});

describe("i18n", () => {
  it("has every key in both languages", () => {
    for (const key of MESSAGE_KEYS) {
      expect(DICTIONARIES.ja[key]).toBeTruthy();
      expect(DICTIONARIES.en[key]).toBeTruthy();
    }
  });
});

describe("multi-frame decoding", () => {
  it("splits a length-prefixed frame sequence", () => {
    const mk = (tick: number, byte: number) => {
      const header = new TextEncoder().encode(JSON.stringify({ tick, width: 8, height: 1, codec: "bitpack1" }));
      const f = new Uint8Array(4 + header.length + 1);
      new DataView(f.buffer).setUint32(0, header.length, false);
      f.set(header, 4);
      f[4 + header.length] = byte;
      return f;
    };
    const a = mk(1, 0b11000000);
    const b = mk(2, 0b11111111);
    const buf = new Uint8Array(4 + 4 + a.length + 4 + b.length);
    const dv = new DataView(buf.buffer);
    dv.setUint32(0, 2, false);
    dv.setUint32(4, a.length, false);
    buf.set(a, 8);
    dv.setUint32(8 + a.length, b.length, false);
    buf.set(b, 12 + a.length);
    const frames = unpackFrames(buf.buffer);
    expect(frames.map((f) => f.header.tick)).toEqual([1, 2]);
    expect(frames.map((f) => popcount(f.bits))).toEqual([2, 8]);
  });
});
