// Phase F12: Live 3D Evolutionary Artificial-Life World (fly_world).
// Starts from a real biological Drosophila connectome as the ancestral brain and evolves brain
// differential genomes, body/leg/wing morphology, and novel food/predator communication signals
// autonomously with ZERO external fitness score!
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../lib/api";
import { useT } from "../i18n";

export interface FlyWorldViewHandle {
  draw: (frame: Frame) => void;
}

interface SignalDictEntry {
  channel: string;
  context: string;
  emit_rate_in_context: number;
  emit_rate_outside: number;
  mi_bits: number;
  receiver_response: number;
}

interface WorldHeader {
  codec: "fly_world_v1";
  tick: number;
  dataset_id: string;
  resolution: string;
  source_cells: number;
  compressed_nodes: number;
  num_flies: number;
  max_population: number;
  arena_half_size: number;
  signal_channel_enabled: boolean;
  predator_enabled: boolean;
  mutations_enabled: boolean;
  mean_generation: number;
  max_generation: number;
  total_births: number;
  total_deaths: number;
  total_predator_kills: number;
  fly_distance: {
    total: number;
    wiring_delta: number;
    morphology_delta: number;
    signal_novelty: number;
    novel_edges: number;
  };
  signal_dictionary: SignalDictEntry[];
  foods: Array<{ x: number; y: number; r: number; amount: number }>;
  predator: { active: boolean; x: number; y: number; r: number };
  feeding: number[];
  recent_births: Array<{ id: number; parents: number[]; gen: number; tick: number; fly_distance: number }>;
}

interface Props {
  live: boolean;
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onDisplayFps?: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

interface EvoFlyRig {
  root: THREE.Group;
  bodyGroup: THREE.Group;
  thoraxMat: THREE.MeshPhysicalMaterial;
  wingL: THREE.Mesh;
  wingR: THREE.Mesh;
  sig0Ring: THREE.Mesh; // Teal food-recruitment signal wave
  sig1Ring: THREE.Mesh; // Magenta predator-alarm signal wave
  feedHalo: THREE.Mesh; // Emerald feeding ring
  legs: THREE.Mesh[];
  phase: number;
}

function makeEvoFlyRig(): EvoFlyRig {
  const root = new THREE.Group();
  const bodyGroup = new THREE.Group();
  root.add(bodyGroup);

  const thoraxMat = new THREE.MeshPhysicalMaterial({
    color: 0xb4772c,
    roughness: 0.38,
    clearcoat: 0.5,
  });
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0xdc2626,
    emissive: 0x450a0a,
    emissiveIntensity: 0.35,
    roughness: 0.2,
    clearcoat: 0.8,
  });
  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xe0f2fe,
    roughness: 0.15,
    transparent: true,
    opacity: 0.52,
    side: THREE.DoubleSide,
  });
  const legMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.55 });

  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.085, 10, 8), thoraxMat);
  thorax.scale.set(1.25, 0.9, 0.85);
  thorax.position.set(0.02, 0, 0.13);
  bodyGroup.add(thorax);

  const abd = new THREE.Mesh(new THREE.SphereGeometry(0.095, 10, 8), thoraxMat);
  abd.scale.set(1.5, 0.9, 0.82);
  abd.position.set(-0.15, 0, 0.115);
  bodyGroup.add(abd);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), thoraxMat);
  head.position.set(0.14, 0, 0.14);
  bodyGroup.add(head);

  for (const sy of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.034, 8, 6), eyeMat);
    eye.position.set(0.16, sy * 0.044, 0.145);
    bodyGroup.add(eye);
  }

  const wingGeo = new THREE.CircleGeometry(0.14, 12);
  wingGeo.scale(1.55, 0.55, 1.0);
  const wingL = new THREE.Mesh(wingGeo, wingMat);
  wingL.position.set(-0.11, 0.065, 0.21);
  wingL.rotation.set(0.08, 0.0, 0.24);
  bodyGroup.add(wingL);

  const wingR = new THREE.Mesh(wingGeo, wingMat);
  wingR.position.set(-0.11, -0.065, 0.21);
  wingR.rotation.set(-0.08, 0.0, -0.24);
  bodyGroup.add(wingR);

  const legs: THREE.Mesh[] = [];
  const legGeo = new THREE.CylinderGeometry(0.011, 0.008, 0.18, 6);
  for (const sx of [0.06, -0.01, -0.08]) {
    for (const sy of [-1, 1]) {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(sx, sy * 0.11, 0.07);
      leg.rotation.x = sy * 0.68;
      bodyGroup.add(leg);
      legs.push(leg);
    }
  }

  // Teal evolved Food-Recruitment Signal wave (sig0)
  const sig0Ring = new THREE.Mesh(
    new THREE.RingGeometry(0.26, 0.36, 24),
    new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  sig0Ring.position.z = 0.018;
  root.add(sig0Ring);

  // Magenta evolved Predator-Alarm Signal wave (sig1)
  const sig1Ring = new THREE.Mesh(
    new THREE.RingGeometry(0.32, 0.45, 24),
    new THREE.MeshBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  sig1Ring.position.z = 0.022;
  root.add(sig1Ring);

  // Emerald feeding halo
  const feedHalo = new THREE.Mesh(
    new THREE.RingGeometry(0.16, 0.23, 20),
    new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  feedHalo.position.z = 0.015;
  root.add(feedHalo);

  return { root, bodyGroup, thoraxMat, wingL, wingR, sig0Ring, sig1Ring, feedHalo, legs, phase: Math.random() * Math.PI * 2 };
}

export const FlyWorldView = forwardRef<FlyWorldViewHandle, Props>(function FlyWorldView(
  { live, onIntervene, onDisplayFps, maxPixels = 1040, testId = "fly-world-view" },
  ref,
) {
  const t = useT();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [hud, setHud] = useState<WorldHeader | null>(null);
  const [renderFps, setRenderFps] = useState(0);
  const [webglStatus, setWebglStatus] = useState<"ok" | "unavailable">("ok");
  const hudThrottleRef = useRef(0);
  const pendingFrameRef = useRef<Frame | null>(null);
  const applyFrameRef = useRef<(frame: Frame) => void>(() => undefined);

  const sceneRef = useRef<{
    scene: THREE.Scene;
    rigs: EvoFlyRig[];
    foodMeshes: THREE.Mesh[];
    predMesh: THREE.Mesh;
    currData: Float32Array | null;
    numFlies: number;
    foods: Array<{ x: number; y: number; r: number; amount: number }>;
    predator: { active: boolean; x: number; y: number; r: number };
    feeding: number[];
  } | null>(null);

  const paneW = Math.max(420, Math.min(1040, maxPixels));
  const paneH = Math.max(360, Math.min(580, Math.round(paneW * 0.54)));

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch {
      setWebglStatus("unavailable");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(paneW, paneH);
    renderer.setClearColor(0x090f1d, 1);
    host.innerHTML = "";
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.up.set(0, 0, 1);
    const camera = new THREE.PerspectiveCamera(42, paneW / paneH, 0.05, 90);
    camera.up.set(0, 0, 1);
    camera.position.set(0.0, -8.5, 8.2);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.15);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    scene.add(new THREE.AmbientLight(0xdbeafe, 0.8));
    const key = new THREE.DirectionalLight(0xfffbeb, 1.35);
    key.position.set(5, -5, 11);
    scene.add(key);

    const half = 5.0;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(half * 2 + 0.6, half * 2 + 0.6),
      new THREE.MeshStandardMaterial({ color: 0x111a2c, roughness: 0.92 }),
    );
    scene.add(floor);
    const grid = new THREE.GridHelper(half * 2, 20, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.004;
    scene.add(grid);

    // Pre-allocate up to 8 dynamic food discs
    const foodMeshes: THREE.Mesh[] = [];
    for (let k = 0; k < 8; k++) {
      const m = new THREE.Mesh(
        new THREE.CircleGeometry(1.0, 32),
        new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          emissive: 0x78350f,
          emissiveIntensity: 0.45,
          roughness: 0.55,
          transparent: true,
          opacity: 0.78,
          side: THREE.DoubleSide,
        }),
      );
      m.position.z = 0.008;
      m.visible = false;
      scene.add(m);
      foodMeshes.push(m);
    }

    // Looming overhead predator shadow disc
    const predMesh = new THREE.Mesh(
      new THREE.CircleGeometry(1.0, 36),
      new THREE.MeshBasicMaterial({
        color: 0x991b1b,
        transparent: true,
        opacity: 0.38,
        side: THREE.DoubleSide,
      }),
    );
    predMesh.position.z = 0.012;
    predMesh.visible = false;
    scene.add(predMesh);

    sceneRef.current = {
      scene,
      rigs: [],
      foodMeshes,
      predMesh,
      currData: null,
      numFlies: 0,
      foods: [],
      predator: { active: false, x: 0, y: 0, r: 1.6 },
      feeding: [],
    };
    if (pendingFrameRef.current) {
      applyFrameRef.current(pendingFrameRef.current);
      pendingFrameRef.current = null;
    }

    let raf = 0;
    let frames = 0;
    let lastMs = performance.now();
    const fpsTimer = window.setInterval(() => {
      setRenderFps(frames);
      onDisplayFps?.(frames);
      frames = 0;
    }, 1000);

    const cAnc = new THREE.Color(0xb4772c); // Natural amber = exact ancestral fly (distance = 0)
    const cMid = new THREE.Color(0x06b6d4); // Cyan = moderate drift
    const cNov = new THREE.Color(0xd946ef); // Magenta = highly diverged artificial-life lineage
    const cTmp = new THREE.Color();

    const animate = () => {
      raf = window.requestAnimationFrame(animate);
      const now = performance.now();
      const dtS = Math.min(0.05, Math.max(0.001, (now - lastMs) * 1e-3));
      lastMs = now;
      const st = sceneRef.current;
      if (st) {
        // Update dynamic regenerating food patches
        for (let k = 0; k < st.foodMeshes.length; k++) {
          const fm = st.foodMeshes[k];
          if (k < st.foods.length && st.foods[k].amount > 0.02) {
            const fd = st.foods[k];
            const rad = fd.r * (0.45 + 0.55 * fd.amount);
            fm.position.set(fd.x, fd.y, 0.008);
            fm.scale.set(rad, rad, 1);
            fm.visible = true;
          } else {
            fm.visible = false;
          }
        }
        // Update looming overhead predator shadow
        st.predMesh.visible = st.predator.active;
        if (st.predator.active) {
          const pulse = 1.0 + 0.06 * Math.sin(now * 0.015);
          st.predMesh.position.set(st.predator.x, st.predator.y, 0.012);
          st.predMesh.scale.set(st.predator.r * pulse, st.predator.r * pulse, 1);
        }

        if (st.currData) {
          const d = st.currData;
          for (let i = 0; i < st.rigs.length; i++) {
            const rig = st.rigs[i];
            if (i >= st.numFlies) {
              rig.root.visible = false;
              continue;
            }
            rig.root.visible = true;
            const b = i * 12;
            const x = d[b + 0];
            const y = d[b + 1];
            const yaw = d[b + 2];
            const spd = d[b + 3];
            const fDist = d[b + 6];
            const bScale = d[b + 7];
            const legScale = d[b + 8];
            const wingScale = d[b + 9];
            const sig0 = d[b + 10];
            const sig1 = d[b + 11];
            const feed = st.feeding[i] ?? 0.0;

            rig.root.position.set(x, y, 0.0);
            rig.root.rotation.z = yaw;
            rig.bodyGroup.scale.set(bScale, bScale, bScale);
            rig.wingL.scale.set(wingScale, wingScale, 1);
            rig.wingR.scale.set(wingScale, wingScale, 1);

            // Color by quantitative distance from ancestral biological fly!
            if (fDist < 0.45) {
              cTmp.copy(cAnc).lerp(cMid, Math.min(1, fDist / 0.45));
            } else {
              cTmp.copy(cMid).lerp(cNov, Math.min(1, (fDist - 0.45) / 0.55));
            }
            rig.thoraxMat.color.copy(cTmp);

            rig.phase += dtS * (4.0 + spd * 12.0);
            for (let li = 0; li < rig.legs.length; li++) {
              rig.legs[li].scale.set(1, legScale, 1);
              rig.legs[li].rotation.y = Math.sin(rig.phase + (li % 2 === 0 ? 0 : Math.PI)) * Math.min(0.45, spd * 0.25);
            }

            // Signal rings expand smoothly while emitting
            const p0 = 1.0 + 0.5 * ((now * 0.003 + i * 0.17) % 1.0);
            rig.sig0Ring.scale.set(p0, p0, 1);
            (rig.sig0Ring.material as THREE.MeshBasicMaterial).opacity = sig0 > 0.2 ? Math.min(0.85, sig0) : 0.0;

            const p1 = 1.0 + 0.65 * ((now * 0.0045 + i * 0.23) % 1.0);
            rig.sig1Ring.scale.set(p1, p1, 1);
            (rig.sig1Ring.material as THREE.MeshBasicMaterial).opacity = sig1 > 0.2 ? Math.min(0.9, sig1) : 0.0;

            (rig.feedHalo.material as THREE.MeshBasicMaterial).opacity = feed > 0.2 ? 0.85 : 0.0;
          }
        }
      }
      controls.update();
      renderer.render(scene, camera);
      frames += 1;
    };
    animate();

    return () => {
      window.cancelAnimationFrame(raf);
      window.clearInterval(fpsTimer);
      controls.dispose();
      renderer.dispose();
      sceneRef.current = null;
    };
  }, [paneW, paneH, onDisplayFps]);

  const applyFrame = (frame: Frame) => {
    try {
      const h = frame.header as unknown as WorldHeader;
      if (!h || h.codec !== "fly_world_v1") return;
      const st = sceneRef.current;
      if (!st) {
        pendingFrameRef.current = frame;
      } else {
        // 4-byte-aligned copy avoids Float32Array RangeError on variable-length JSON headers!
        const byteLen = Math.floor(frame.bits.byteLength / 4) * 4;
        const aligned = new Uint8Array(byteLen);
        aligned.set(frame.bits.subarray(0, byteLen));
        const copy = new Float32Array(aligned.buffer);
        const count = Math.min(h.num_flies ?? 0, Math.floor(copy.length / 12));
        while (st.rigs.length < count) {
          const rig = makeEvoFlyRig();
          st.scene.add(rig.root);
          st.rigs.push(rig);
        }
        st.numFlies = count;
        st.currData = copy;
        st.foods = Array.isArray(h.foods) ? h.foods : [];
        st.predator = h.predator ?? { active: false, x: 0, y: 0, r: 1.6 };
        st.feeding = Array.isArray(h.feeding) ? h.feeding : [];
      }
      const nowHud = performance.now();
      if (hudThrottleRef.current === 0 || nowHud - hudThrottleRef.current > 100) {
        hudThrottleRef.current = nowHud;
        setHud(h);
      }
    } catch (err) {
      console.error("FlyWorldView draw error:", err);
    }
  };
  applyFrameRef.current = applyFrame;

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => applyFrameRef.current(frame),
  }));

  return (
    <div data-testid={testId} style={{ display: "flex", flexDirection: "column", gap: "10px", width: "100%" }}>
      <div className="notice small" style={{ margin: 0, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px", borderLeft: "3px solid #d946ef" }}>
        <span>{t("flyworld.banner")}</span>
        {hud && (
          <span className="mono">
            {hud.dataset_id} ({(hud.source_cells ?? 0).toLocaleString()} → {(hud.compressed_nodes ?? 0).toLocaleString()} nodes) · {renderFps} FPS
          </span>
        )}
      </div>

      {/* Live Evolutionary Interventions */}
      <div className="toolbar" role="group" style={{ margin: 0 }}>
        <button type="button" disabled={!live} onClick={() => onIntervene("toggle_signals", {})}>
          {hud?.signal_channel_enabled ? `📡 ${t("flyworld.act.signals_off")}` : `📡 ${t("flyworld.act.signals_on")}`}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("spawn_predator", {})}>
          🦅 {t("flyworld.act.spawn_predator")}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("add_food", {})}>
          🍯 {t("flyworld.act.add_food")}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("toggle_mutations", {})}>
          {hud?.mutations_enabled ? `🧬 ${t("flyworld.act.mut_off")}` : `🧬 ${t("flyworld.act.mut_on")}`}
        </button>
      </div>

      {/* 3D Evolutionary World Canvas */}
      <div style={{ border: "1px solid var(--border, #334155)", borderRadius: "8px", overflow: "hidden", background: "#090f1d" }}>
        <div style={{ padding: "6px 10px", fontSize: "0.82rem", display: "flex", justifyContent: "space-between", background: "rgba(15,23,42,0.92)", color: "#e2e8f0", borderBottom: "1px solid #1e293b" }}>
          <strong>{t("flyworld.arena_title")}</strong>
          <span className="mono" style={{ opacity: 0.85 }}>
            {hud ? `tick ${hud.tick ?? 0} · Gen max ${hud.max_generation ?? 0} (mean ${(hud.mean_generation ?? 0).toFixed(1)})` : ""}
          </span>
        </div>
        {webglStatus === "unavailable" ? (
          <div style={{ padding: "24px", color: "#94a3b8", textAlign: "center" }}>WebGL unavailable</div>
        ) : (
          <div ref={hostRef} style={{ width: "100%", height: `${paneH}px`, overflow: "hidden" }} />
        )}
      </div>

      {/* Live Evolutionary Telemetry & Distance From Ancestral Fly */}
      {hud && (
        <>
          <div className="panel" style={{ margin: 0, padding: "10px 12px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "10px", fontSize: "0.84rem" }}>
            <div>
              <div className="muted small">{t("metric.population")}</div>
              <strong className="mono" style={{ fontSize: "1.05rem", color: "#34d399" }}>
                {hud.num_flies ?? 0} / {hud.max_population ?? 48}
              </strong>
            </div>
            <div>
              <div className="muted small">{t("metric.total_births")} / {t("metric.total_deaths")}</div>
              <strong className="mono" style={{ fontSize: "1.05rem" }}>
                +{hud.total_births ?? 0} / -{hud.total_deaths ?? 0} (🦅 {hud.total_predator_kills ?? 0})
              </strong>
            </div>
            <div>
              <div className="muted small">{t("metric.fly_distance")}</div>
              <strong className="mono" style={{ fontSize: "1.05rem", color: "#d946ef" }}>
                {(hud.fly_distance?.total ?? 0).toFixed(3)}
              </strong>
            </div>
            <div>
              <div className="muted small">{t("flyworld.dist_breakdown")}</div>
              <strong className="mono" style={{ fontSize: "0.9rem", color: "#38bdf8" }}>
                W:{(hud.fly_distance?.wiring_delta ?? 0).toFixed(2)} M:{(hud.fly_distance?.morphology_delta ?? 0).toFixed(2)} S:{(hud.fly_distance?.signal_novelty ?? 0).toFixed(2)}
              </strong>
            </div>
          </div>

          {/* Evolved Signal Dictionary */}
          <div className="panel" style={{ margin: 0, padding: "10px 12px", fontSize: "0.82rem" }}>
            <strong style={{ display: "block", marginBottom: "6px" }}>{t("flyworld.signal_dict_title")}</strong>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid #334155" }}>
                  <th style={{ padding: "4px" }}>{t("flyworld.col.channel")}</th>
                  <th style={{ padding: "4px" }}>{t("flyworld.col.context")}</th>
                  <th style={{ padding: "4px" }}>{t("flyworld.col.emit_in_out")}</th>
                  <th style={{ padding: "4px" }}>MI (bits)</th>
                  <th style={{ padding: "4px" }}>{t("flyworld.col.recv")}</th>
                </tr>
              </thead>
              <tbody className="mono">
                {(hud.signal_dictionary ?? []).map((row) => (
                  <tr key={row.channel} style={{ borderBottom: "1px solid #1e293b" }}>
                    <td style={{ padding: "4px", color: row.channel === "sig0_food" ? "#2dd4bf" : "#f43f5e" }}>
                      {row.channel}
                    </td>
                    <td style={{ padding: "4px" }}>{row.context}</td>
                    <td style={{ padding: "4px" }}>
                      {((row.emit_rate_in_context ?? 0) * 100).toFixed(1)}% vs {((row.emit_rate_outside ?? 0) * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "4px", fontWeight: 700 }}>{(row.mi_bits ?? 0).toFixed(4)}</td>
                    <td style={{ padding: "4px" }}>{(row.receiver_response ?? 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
});
