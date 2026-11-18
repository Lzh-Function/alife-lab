import { describe, expect, it } from "vitest";
import { bitAt, circleCells, unpackFrame } from "./api";
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
