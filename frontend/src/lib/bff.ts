// Shared BFF display helpers (instruction colours and disassembly).
export const OP_COLORS: Record<number, [number, number, number]> = {
  60: [80, 160, 255], // <
  62: [40, 120, 230], // >
  123: [180, 120, 255], // {
  125: [140, 80, 230], // }
  43: [60, 220, 120], // +
  45: [30, 160, 90], // -
  46: [255, 210, 60], // .
  44: [255, 160, 40], // ,
  91: [255, 90, 90], // [
  93: [220, 50, 50], // ]
};

export const OP_NAMES: Record<number, string> = {
  60: "head0 −1",
  62: "head0 +1",
  123: "head1 −1",
  125: "head1 +1",
  43: "tape[head0] +1",
  45: "tape[head0] −1",
  46: "tape[head1] ← tape[head0]",
  44: "tape[head0] ← tape[head1]",
  91: "if tape[head0] == 0: jump past matching ]",
  93: "if tape[head0] != 0: jump back to matching [",
};

/** One printable character per byte: instructions as themselves, NUL as '0', others '·'. */
export function glyph(b: number): string {
  return OP_COLORS[b] ? String.fromCharCode(b) : b === 0 ? "0" : "·";
}

export function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return out;
}
