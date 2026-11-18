import { useState } from "react";
import { useT } from "../i18n";
import { api, ApiError } from "../lib/api";
import { BffDebugger, type BffTrace } from "./BffDebugger";

// Stand-alone BFF VM debugger: trace two programs without a run (programs are data only).
export function BffDebuggerPage() {
  const t = useT();
  // Default: a hand-written 7-byte copier for the `bff` variant (head0 = 0x80 mod 128 = 0,
  // head1 = 0xC0 mod 128 = 64, then "[.>}]" copies bytes while tape[head0] != 0).
  const [a, setA] = useState("80c05b2e3e7d5d");
  const [b, setB] = useState("");
  const [encoding, setEncoding] = useState<"text" | "hex">("hex");
  const [variant, setVariant] = useState("bff");
  const [limit, setLimit] = useState(8192);
  const [trace, setTrace] = useState<BffTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = () =>
    api
      .bffTrace({ program_a: a, program_b: b, encoding, variant, step_limit: limit })
      .then((tr) => {
        setTrace(tr);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));

  return (
    <div>
      <h2>{t("debugger.page_title")}</h2>
      <p className="muted small">{t("debugger.page_note")}</p>
      <p className="muted small">{t("debugger.default_note")}</p>
      <div className="panel">
        <label className="block">
          {t("debugger.program_a")}
          <input className="mono" data-testid="dbg-program-a" value={a} maxLength={encoding === "hex" ? 128 : 64} onChange={(e) => setA(e.target.value)} style={{ width: "100%" }} />
        </label>
        <label className="block">
          {t("debugger.program_b")}
          <input className="mono" data-testid="dbg-program-b" value={b} maxLength={encoding === "hex" ? 128 : 64} onChange={(e) => setB(e.target.value)} style={{ width: "100%" }} />
        </label>
        <div className="toolbar">
          <label className="inline">
            {t("debugger.encoding")}
            <select value={encoding} onChange={(e) => setEncoding(e.target.value as "text" | "hex")}>
              <option value="text">text (latin-1)</option>
              <option value="hex">hex</option>
            </select>
          </label>
          <label className="inline">
            {t("form.bff.variant")}
            <select value={variant} onChange={(e) => setVariant(e.target.value)}>
              <option value="bff_noheads">bff_noheads</option>
              <option value="bff">bff</option>
            </select>
          </label>
          <label className="inline">
            {t("form.bff.step_limit")}
            <input type="number" min={0} max={8192} value={limit} onChange={(e) => setLimit(Number(e.target.value))} style={{ width: "6rem" }} />
          </label>
          <button className="primary" data-testid="dbg-run" onClick={run}>
            {t("debugger.run")}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>
      {trace && <BffDebugger trace={trace} />}
    </div>
  );
}
