// Phase F11 Stage 3: Live 3D Collective Whole-Brain Fly Swarm Renderer (10-64 flies in ONE arena).
// Renders all N flies at 60 FPS with snapshot interpolation, animated legs/wings, glowing neural
// arousal halos, golden acoustic song rings, cyan leg-touch encounter sparks, and aversive odor mist.
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../lib/api";
import { useT } from "../i18n";

export interface FlySwarmViewHandle {
  draw: (frame: Frame) => void;
}

interface SwarmHeader {
  codec: "fly_swarm_v1";
  tick: number;
  dataset_id: string;
  wiring: string;
  social_mode: string;
  num_flies: number;
  cells_per_fly: number;
  total_neurons: number;
  arena_half_size: number;
  odor_enabled: boolean;
  enable_touch: boolean;
  clean_zone_fraction: number;
  encounter_rate: number;
  total_encounters: number;
  mean_speed: number;
  active_singers: number;
  total_lunges?: number;
  active_threats?: number;
  mean_da1_pn_hz?: number;
  mean_hg34_hz?: number;
  realtime_factor: number;
  contacts: Array<[number, number, number, number]>;
  clashes?: Array<[number, number, number, number]>;
}

interface Props {
  live: boolean;
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onDisplayFps?: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

interface FlyMeshRig {
  root: THREE.Group;
  wingL: THREE.Mesh;
  wingR: THREE.Mesh;
  songRing: THREE.Mesh;
  touchHalo: THREE.Mesh;
  threatHalo: THREE.Mesh;
  legs: THREE.Mesh[];
  phase: number;
}

function makeFlyRig(isMale: boolean): FlyMeshRig {
  const root = new THREE.Group();

  const thoraxMat = new THREE.MeshPhysicalMaterial({
    color: isMale ? 0xb4772c : 0x9c6628,
    roughness: 0.4,
    clearcoat: 0.45,
  });
  const abdMat = new THREE.MeshPhysicalMaterial({
    color: isMale ? 0x1e293b : 0xd49c3d,
    roughness: 0.35,
    clearcoat: 0.55,
  });
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0xdc2626,
    emissive: 0x450a0a,
    emissiveIntensity: 0.4,
    roughness: 0.18,
    clearcoat: 0.85,
  });
  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xe0f2fe,
    roughness: 0.15,
    transparent: true,
    opacity: 0.48,
    side: THREE.DoubleSide,
  });
  const legMat = new THREE.MeshStandardMaterial({ color: 0x5c3d1e, roughness: 0.6 });

  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.085, 12, 10), thoraxMat);
  thorax.scale.set(1.25, 0.9, 0.85);
  thorax.position.set(0.02, 0, 0.13);
  thorax.castShadow = true;
  root.add(thorax);

  const abd = new THREE.Mesh(new THREE.SphereGeometry(0.095, 12, 10), abdMat);
  abd.scale.set(1.55, 0.92, 0.82);
  abd.position.set(-0.15, 0, 0.115);
  abd.castShadow = true;
  root.add(abd);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.062, 10, 8), thoraxMat);
  head.position.set(0.145, 0, 0.14);
  root.add(head);

  for (const sy of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.036, 10, 8), eyeMat);
    eye.position.set(0.165, sy * 0.045, 0.145);
    root.add(eye);
  }

  const wingGeo = new THREE.CircleGeometry(0.14, 14);
  wingGeo.scale(1.55, 0.55, 1.0);
  const wingL = new THREE.Mesh(wingGeo, wingMat);
  wingL.position.set(-0.12, 0.065, 0.21);
  wingL.rotation.set(0.08, 0.0, 0.24);
  root.add(wingL);

  const wingR = new THREE.Mesh(wingGeo, wingMat);
  wingR.position.set(-0.12, -0.065, 0.21);
  wingR.rotation.set(-0.08, 0.0, -0.24);
  root.add(wingR);

  const legs: THREE.Mesh[] = [];
  const legGeo = new THREE.CylinderGeometry(0.011, 0.008, 0.18, 6);
  for (const sx of [0.06, -0.01, -0.08]) {
    for (const sy of [-1, 1]) {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(sx, sy * 0.11, 0.07);
      leg.rotation.x = sy * 0.68;
      root.add(leg);
      legs.push(leg);
    }
  }

  // Golden acoustic wing-song halo (visible when courting male sings)
  const songRing = new THREE.Mesh(
    new THREE.RingGeometry(0.24, 0.34, 24),
    new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  songRing.position.z = 0.015;
  root.add(songRing);

  // Cyan mechanosensory leg-touch spark ring (visible during Ramdya encounter cascade!)
  const touchHalo = new THREE.Mesh(
    new THREE.RingGeometry(0.19, 0.26, 20),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  touchHalo.position.z = 0.02;
  root.add(touchHalo);

  // Crimson bilateral wing-threat halo (visible during male-male aggression!)
  const threatHalo = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.34, 24),
    new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.0, side: THREE.DoubleSide }),
  );
  threatHalo.position.z = 0.025;
  root.add(threatHalo);

  return { root, wingL, wingR, songRing, touchHalo, threatHalo, legs, phase: Math.random() * Math.PI * 2 };
}

export const FlySwarmView = forwardRef<FlySwarmViewHandle, Props>(function FlySwarmView(
  { live, onIntervene, onDisplayFps, maxPixels = 980, testId = "fly-swarm-view" },
  ref,
) {
  const t = useT();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [hud, setHud] = useState<SwarmHeader | null>(null);
  const [renderFps, setRenderFps] = useState(0);
  const [webglStatus, setWebglStatus] = useState<"ok" | "unavailable">("ok");

  const sceneRef = useRef<{
    scene: THREE.Scene;
    arenaHalf: number;
    rigs: FlyMeshRig[];
    odorZone: THREE.Mesh | null;
    divLine: THREE.Line | null;
    foodPatch: THREE.Mesh | null;
    contactLines: THREE.LineSegments | null;
    clashLines: THREE.LineSegments | null;
    prevData: Float32Array | null;
    currData: Float32Array | null;
    numFlies: number;
    socialMode: string;
    lastFrameMs: number;
    frameIntervalMs: number;
    odorEnabled: boolean;
    contacts: Array<[number, number, number, number]>;
    clashes: Array<[number, number, number, number]>;
  } | null>(null);

  const hudThrottleRef = useRef(0);
  const pendingFrameRef = useRef<Frame | null>(null);
  const applyFrameRef = useRef<(frame: Frame) => void>(() => undefined);

  const paneW = Math.max(420, Math.min(1040, maxPixels));
  const paneH = Math.max(360, Math.min(600, Math.round(paneW * 0.56)));

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
    renderer.shadowMap.enabled = true;
    renderer.setClearColor(0x0b1220, 1);
    host.innerHTML = "";
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.up.set(0, 0, 1);
    const camera = new THREE.PerspectiveCamera(42, paneW / paneH, 0.05, 80);
    camera.up.set(0, 0, 1);
    camera.position.set(0.0, -7.8, 7.2);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.15);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    scene.add(new THREE.AmbientLight(0xdbeafe, 0.75));
    const key = new THREE.DirectionalLight(0xfffbeb, 1.35);
    key.position.set(4, -5, 10);
    key.castShadow = true;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x93c5fd, 0.45);
    fill.position.set(-5, 4, 6);
    scene.add(fill);

    // Arena petri dish floor + center divider (left = aversive odor, right = clean air)
    const half = 4.5;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(half * 2 + 0.6, half * 2 + 0.6),
      new THREE.MeshStandardMaterial({ color: 0x162032, roughness: 0.92 }),
    );
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(half * 2, 18, 0x3b485e, 0x222d40);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.003;
    scene.add(grid);

    // Left-half aversive odor mist overlay (x < 0)
    const odorZone = new THREE.Mesh(
      new THREE.PlaneGeometry(half, half * 2),
      new THREE.MeshBasicMaterial({
        color: 0xa855f7,
        transparent: true,
        opacity: 0.16,
        side: THREE.DoubleSide,
      }),
    );
    odorZone.position.set(-half / 2, 0, 0.006);
    scene.add(odorZone);

    // Crisp divider line at x = 0 separating aversive odor (left) vs clean air (right)
    const divLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -half, 0.01),
        new THREE.Vector3(0, half, 0.01),
      ]),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.7 }),
    );
    scene.add(divLine);

    // Territorial central yeast/food patch (visible in male_aggression_arena mode)
    const foodPatch = new THREE.Mesh(
      new THREE.CircleGeometry(0.85, 32),
      new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0x78350f,
        emissiveIntensity: 0.45,
        roughness: 0.65,
        side: THREE.DoubleSide,
      }),
    );
    foodPatch.position.set(0, 0, 0.009);
    foodPatch.visible = false;
    scene.add(foodPatch);

    // Contact spark segments between touching flies
    const maxSegs = 64;
    const contactGeo = new THREE.BufferGeometry();
    contactGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxSegs * 2 * 3), 3));
    const contactLines = new THREE.LineSegments(
      contactGeo,
      new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.9 }),
    );
    contactLines.frustumCulled = false;
    scene.add(contactLines);

    // Crimson lunge/clash arcs between dueling males
    const clashGeo = new THREE.BufferGeometry();
    clashGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxSegs * 2 * 3), 3));
    const clashLines = new THREE.LineSegments(
      clashGeo,
      new THREE.LineBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.95 }),
    );
    clashLines.frustumCulled = false;
    scene.add(clashLines);

    sceneRef.current = {
      scene,
      arenaHalf: half,
      rigs: [],
      odorZone,
      divLine,
      foodPatch,
      contactLines,
      clashLines,
      prevData: null,
      currData: null,
      numFlies: 0,
      socialMode: "ramdya_odor_avoidance",
      lastFrameMs: performance.now(),
      frameIntervalMs: 16.6,
      odorEnabled: true,
      contacts: [],
      clashes: [],
    };
    if (pendingFrameRef.current) {
      applyFrameRef.current(pendingFrameRef.current);
      pendingFrameRef.current = null;
    }

    let raf = 0;
    let frames = 0;
    let lastTickMs = performance.now();
    const fpsTimer = window.setInterval(() => {
      setRenderFps(frames);
      onDisplayFps?.(frames);
      frames = 0;
    }, 1000);

    const animate = () => {
      raf = window.requestAnimationFrame(animate);
      const now = performance.now();
      const dtS = Math.min(0.05, Math.max(0.001, (now - lastTickMs) * 1e-3));
      lastTickMs = now;
      const st = sceneRef.current;
      if (st && st.currData && st.rigs.length === st.numFlies) {
        const alpha = st.prevData
          ? Math.min(1.0, Math.max(0.0, (now - st.lastFrameMs) / Math.max(8, st.frameIntervalMs)))
          : 1.0;
        const curr = st.currData;
        const prev = st.prevData ?? curr;

        const isAgg = st.socialMode === "male_aggression_arena";
        if (st.odorZone) st.odorZone.visible = !isAgg && st.odorEnabled;
        if (st.divLine) st.divLine.visible = !isAgg;
        if (st.foodPatch) st.foodPatch.visible = isAgg;

        for (let i = 0; i < st.numFlies; i++) {
          const b = i * 8;
          const x = prev[b + 0] + (curr[b + 0] - prev[b + 0]) * alpha;
          const y = prev[b + 1] + (curr[b + 1] - prev[b + 1]) * alpha;
          let dyaw = curr[b + 2] - prev[b + 2];
          if (dyaw > Math.PI) dyaw -= 2 * Math.PI;
          if (dyaw < -Math.PI) dyaw += 2 * Math.PI;
          const yaw = prev[b + 2] + dyaw * alpha;
          const spd = curr[b + 3];
          const touchOrLunge = curr[b + 5];
          const song = curr[b + 6];
          const threat = isAgg ? curr[b + 7] : 0.0;

          const rig = st.rigs[i];
          rig.root.position.set(x, y, 0.0);
          rig.root.rotation.z = yaw;

          // Animate 6 legs smoothly with walking speed
          rig.phase += dtS * (4.0 + spd * 14.0);
          for (let li = 0; li < rig.legs.length; li++) {
            const amp = Math.min(0.45, spd * 0.25);
            rig.legs[li].rotation.y = Math.sin(rig.phase + (li % 2 === 0 ? 0 : Math.PI)) * amp;
          }

          if (isAgg && threat > 0.12) {
            // Bilateral wing-threat display (hg3/hg4 MN): BOTH wings elevated & splayed symmetrically!
            const elev = Math.min(0.95, 0.35 + threat * 0.65);
            rig.wingL.rotation.set(0.55 * elev, -0.45 * elev, 0.24 + 0.65 * elev);
            rig.wingR.rotation.set(-0.55 * elev, -0.45 * elev, -0.24 - 0.65 * elev);
          } else {
            // Unilateral courtship wing splay or resting posture
            const splay = song > 0.1 ? 0.55 + 0.18 * Math.sin(now * 0.06) : 0.0;
            rig.wingL.rotation.set(0.08, 0.0, 0.24 + splay);
            rig.wingR.rotation.set(-0.08, 0.0, -0.24 - splay * 0.25);
          }
          (rig.songRing.material as THREE.MeshBasicMaterial).opacity = song > 0.1 ? Math.min(0.75, song * 0.8) : 0.0;

          // Cyan leg-touch halo (Ramdya) OR crimson threat/lunge halo (Aggression)
          (rig.touchHalo.material as THREE.MeshBasicMaterial).opacity = !isAgg && touchOrLunge > 0.5 ? 0.85 : 0.0;
          const threatOpacity = isAgg
            ? Math.min(0.92, (touchOrLunge > 0.5 ? 0.88 : threat * 0.85) * (0.75 + 0.25 * Math.sin(now * 0.02)))
            : 0.0;
          (rig.threatHalo.material as THREE.MeshBasicMaterial).opacity = threatOpacity;
        }

        // Update cyan contact spark lines between touching flies
        if (st.contactLines) {
          const posAttr = st.contactLines.geometry.getAttribute("position") as THREE.BufferAttribute;
          const arr = posAttr.array as Float32Array;
          const cnt = Math.min(64, st.contacts.length);
          for (let c = 0; c < cnt; c++) {
            const [x1, y1, x2, y2] = st.contacts[c];
            const k = c * 6;
            arr[k + 0] = x1;
            arr[k + 1] = y1;
            arr[k + 2] = 0.14;
            arr[k + 3] = x2;
            arr[k + 4] = y2;
            arr[k + 5] = 0.14;
          }
          st.contactLines.geometry.setDrawRange(0, cnt * 2);
          posAttr.needsUpdate = true;
        }
        // Update crimson-orange clash arcs between lunging males
        if (st.clashLines) {
          const posAttr = st.clashLines.geometry.getAttribute("position") as THREE.BufferAttribute;
          const arr = posAttr.array as Float32Array;
          const cnt = Math.min(64, st.clashes.length);
          for (let c = 0; c < cnt; c++) {
            const [x1, y1, x2, y2] = st.clashes[c];
            const k = c * 6;
            arr[k + 0] = x1;
            arr[k + 1] = y1;
            arr[k + 2] = 0.24;
            arr[k + 3] = x2;
            arr[k + 4] = y2;
            arr[k + 5] = 0.24;
          }
          st.clashLines.geometry.setDrawRange(0, cnt * 2);
          posAttr.needsUpdate = true;
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
      const h = frame.header as unknown as SwarmHeader;
      if (!h || h.codec !== "fly_swarm_v1") return;
      const st = sceneRef.current;
      if (!st) {
        pendingFrameRef.current = frame;
      } else {
        const now = performance.now();
        const dt = now - st.lastFrameMs;
        if (dt > 2 && dt < 250) st.frameIntervalMs = 0.7 * st.frameIntervalMs + 0.3 * dt;
        st.lastFrameMs = now;

        // Copy payload into a 4-byte-aligned buffer: WebSocket frame.bits starts right after a
        // variable-length JSON header so frame.bits.byteOffset is often NOT a multiple of 4.
        const byteLen = Math.floor(frame.bits.byteLength / 4) * 4;
        const aligned = new Uint8Array(byteLen);
        aligned.set(frame.bits.subarray(0, byteLen));
        const copy = new Float32Array(aligned.buffer);
        const count = Math.min(h.num_flies ?? 0, Math.floor(copy.length / 8));
        if (st.numFlies !== count) {
          for (const r of st.rigs) st.scene.remove(r.root);
          st.rigs = [];
          for (let i = 0; i < count; i++) {
            const isMale = copy[i * 8 + 4] > 0.5;
            const rig = makeFlyRig(isMale);
            st.scene.add(rig.root);
            st.rigs.push(rig);
          }
          st.numFlies = count;
          st.prevData = copy;
          st.currData = copy;
        } else {
          st.prevData = st.currData;
          st.currData = copy;
        }
        st.socialMode = h.social_mode ?? "ramdya_odor_avoidance";
        st.odorEnabled = Boolean(h.odor_enabled);
        st.contacts = Array.isArray(h.contacts) ? h.contacts : [];
        st.clashes = Array.isArray(h.clashes) ? h.clashes : [];
      }
      const nowHud = performance.now();
      if (hudThrottleRef.current === 0 || nowHud - hudThrottleRef.current > 90) {
        hudThrottleRef.current = nowHud;
        setHud(h);
      }
    } catch (err) {
      console.error("FlySwarmView draw error:", err);
    }
  };
  applyFrameRef.current = applyFrame;

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => applyFrameRef.current(frame),
  }));

  return (
    <div data-testid={testId} style={{ display: "flex", flexDirection: "column", gap: "10px", width: "100%" }}>
      <div className="notice small" style={{ margin: 0, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px", borderLeft: "3px solid #38bdf8" }}>
        <span>{t("swarm.banner")}</span>
        {hud && (
          <span className="mono">
            {hud.num_flies ?? 0} flies · {(hud.total_neurons ?? 0).toLocaleString()} neurons · {(hud.realtime_factor ?? 0).toFixed(2)}x RT · {renderFps} FPS
          </span>
        )}
      </div>

      {/* Live Swarm Interventions */}
      <div className="toolbar" role="group" style={{ margin: 0 }}>
        <button type="button" disabled={!live} onClick={() => onIntervene("toggle_odor", {})}>
          {hud?.social_mode === "male_aggression_arena"
            ? hud?.odor_enabled
              ? `🧪 ${t("swarm.act.cva_off")}`
              : `🧪 ${t("swarm.act.cva_on")}`
            : hud?.odor_enabled
              ? `🟣 ${t("swarm.act.odor_off")}`
              : `🟣 ${t("swarm.act.odor_on")}`}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("toggle_touch", {})}>
          {hud?.social_mode === "male_aggression_arena"
            ? hud?.enable_touch
              ? `👁 ${t("swarm.act.vision_off")}`
              : `👁 ${t("swarm.act.vision_on")}`
            : hud?.enable_touch
              ? `⚡ ${t("swarm.act.touch_off")}`
              : `⚡ ${t("swarm.act.touch_on")}`}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("scatter", {})}>
          🔄 {t("swarm.act.scatter")}
        </button>
        <button type="button" disabled={!live} onClick={() => onIntervene("wake_all", {})}>
          🔔 {t("swarm.act.wake_all")}
        </button>
      </div>

      {/* 3D Petri Dish Swarm Canvas */}
      <div style={{ border: "1px solid var(--border, #334155)", borderRadius: "8px", overflow: "hidden", background: "#0b1220" }}>
        <div style={{ padding: "6px 10px", fontSize: "0.82rem", display: "flex", justifyContent: "space-between", background: "rgba(15,23,42,0.92)", color: "#e2e8f0", borderBottom: "1px solid #1e293b" }}>
          <strong>{hud?.social_mode === "male_aggression_arena" ? t("swarm.aggression_title") : t("swarm.arena_title")}</strong>
          <span className="mono" style={{ opacity: 0.85 }}>
            {hud ? `${hud.dataset_id ?? ""} (${hud.wiring ?? ""}) · tick ${hud.tick ?? 0}` : ""}
          </span>
        </div>
        {webglStatus === "unavailable" ? (
          <div style={{ padding: "24px", color: "#94a3b8", textAlign: "center" }}>WebGL unavailable</div>
        ) : (
          <div ref={hostRef} style={{ width: "100%", height: `${paneH}px`, overflow: "hidden" }} />
        )}
      </div>

      {/* Live Collective Metrics Panel */}
      {hud && hud.social_mode === "male_aggression_arena" ? (
        <div className="panel" style={{ margin: 0, padding: "10px 12px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px", fontSize: "0.84rem" }}>
          <div>
            <div className="muted small">{t("metric.total_lunges")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#f97316" }}>
              {hud.total_lunges ?? 0}
            </strong>
          </div>
          <div>
            <div className="muted small">{t("metric.active_threats")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#ef4444" }}>
              {hud.active_threats ?? 0} / {hud.num_flies ?? 0}
            </strong>
          </div>
          <div>
            <div className="muted small">DA1_PN (cVA)</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#a78bfa" }}>
              {(hud.mean_da1_pn_hz ?? 0).toFixed(1)} Hz
            </strong>
          </div>
          <div>
            <div className="muted small">hg3/hg4 MN (Wing Threat)</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#fbbf24" }}>
              {(hud.mean_hg34_hz ?? 0).toFixed(1)} Hz
            </strong>
          </div>
        </div>
      ) : hud && (
        <div className="panel" style={{ margin: 0, padding: "10px 12px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px", fontSize: "0.84rem" }}>
          <div>
            <div className="muted small">{t("metric.clean_zone_fraction")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#34d399" }}>
              {((hud.clean_zone_fraction ?? 0) * 100).toFixed(1)}%
            </strong>
          </div>
          <div>
            <div className="muted small">{t("metric.encounter_rate")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#38bdf8" }}>
              {(hud.encounter_rate ?? 0).toFixed(2)} /s ({hud.total_encounters ?? 0})
            </strong>
          </div>
          <div>
            <div className="muted small">{t("metric.mean_speed_cm_s")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem" }}>{(hud.mean_speed ?? 0).toFixed(2)} cm/s</strong>
          </div>
          <div>
            <div className="muted small">{t("metric.active_singers")}</div>
            <strong className="mono" style={{ fontSize: "1.1rem", color: "#fbbf24" }}>{hud.active_singers ?? 0}</strong>
          </div>
        </div>
      )}
    </div>
  );
});
