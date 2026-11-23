// 3D glowing whole-brain point cloud (Three.js / WebGL2, presentation-only).
// Static 3D FAFB coordinates for all N neurons (up to 138,639 in FlyWire v783) are fetched once
// via GET /api/v1/connectome/{dataset_id}/geometry and cached in memory. Per-tick connectome_v1
// frames carry only the fired neuron indices (int32), lighting up spiking cells in real time.
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api, type ConnectomeFrameHeader, type ConnectomeGeometry, type Frame } from "../lib/api";
import { useT } from "../i18n";

export interface Brain3DHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  datasetId: string;
  onDisplayFps?: (fps: number) => void;
  testId?: string;
  width?: number;
  source?: string;
}

const GEOM_CACHE = new Map<string, Promise<ConnectomeGeometry>>();

function fetchGeometryCached(datasetId: string): Promise<ConnectomeGeometry> {
  let p = GEOM_CACHE.get(datasetId);
  if (!p) {
    p = api.connectomeGeometry(datasetId).catch((err) => {
      GEOM_CACHE.delete(datasetId);
      throw err;
    });
    GEOM_CACHE.set(datasetId, p);
  }
  return p;
}

// Role palette: sensory = sky cyan, interneuron = cool slate-blue, motor = coral rose, other = lavender.
const ROLE_COLORS: Array<[number, number, number]> = [
  [0.22, 0.74, 0.97],
  [0.36, 0.50, 0.72],
  [0.98, 0.44, 0.52],
  [0.65, 0.55, 0.98],
];

function regionColor(idx: number): [number, number, number] {
  if (idx === 255) return [0.35, 0.42, 0.55];
  const hue = (idx * 0.61803398875) % 1.0;
  const c = new THREE.Color().setHSL(hue, 0.65, 0.55);
  return [c.r, c.g, c.b];
}

const VERTEX_SHADER = `
attribute vec3 aColor;
attribute float aFireTime;
uniform float uBaseSize;
uniform float uPixelRatio;
uniform float uTime;
uniform float uDecayRate;
varying vec3 vColor;
varying float vGlow;

void main() {
  vColor = aColor;
  float age = max(0.0, uTime - aFireTime);
  vGlow = aFireTime < -100.0 ? 0.0 : clamp(exp(-age * uDecayRate), 0.0, 1.0);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  float dist = max(0.35, -mvPosition.z);
  float size = uBaseSize * (1.0 + vGlow * 3.4) * uPixelRatio / dist;
  gl_PointSize = clamp(size, 1.2, 26.0);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const FRAGMENT_SHADER = `
varying vec3 vColor;
varying float vGlow;

void main() {
  vec2 uv = gl_PointCoord - vec2(0.5);
  float d = length(uv);
  if (d > 0.5) discard;
  float core = smoothstep(0.5, 0.05, d);
  vec3 hot = vec3(1.0, 0.93, 0.55);
  vec3 col = mix(vColor, hot, clamp(vGlow * 1.15, 0.0, 1.0));
  float alpha = mix(0.24, 0.98, clamp(vGlow, 0.0, 1.0)) * core;
  gl_FragColor = vec4(col * (1.0 + vGlow * 1.2), alpha);
}
`;

export const Brain3D = forwardRef<Brain3DHandle, Props>(function Brain3D(
  { datasetId, onDisplayFps, testId = "world-canvas", width = 800, source = "live" },
  ref,
) {
  const t = useT();
  const hostRef = useRef<HTMLDivElement>(null);
  const [colorMode, setColorMode] = useState<"role" | "region">("role");
  const [autoRotate, setAutoRotate] = useState(true);
  const [status, setStatus] = useState<string>("loading");
  const [cellCount, setCellCount] = useState<number>(0);
  const [activeCount, setActiveCount] = useState<number>(0);

  const colorModeRef = useRef(colorMode);
  colorModeRef.current = colorMode;
  const autoRotateRef = useRef(autoRotate);
  autoRotateRef.current = autoRotate;
  const sourceRef = useRef(source);
  sourceRef.current = source;

  const engineRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    geom: ConnectomeGeometry | null;
    points: THREE.Points | null;
    fireTimeAttr: THREE.BufferAttribute | null;
    colorAttr: THREE.BufferAttribute | null;
    fireTimeArr: Float32Array | null;
    timeUniform: { value: number } | null;
    decayUniform: { value: number } | null;
    pendingFrame: Frame | null;
    applyColors: (mode: "role" | "region") => void;
  } | null>(null);
  const framesRef = useRef(0);
  const lastCountUiRef = useRef(0);
  const startTimeRef = useRef(performance.now());

  const height = Math.round(width * 0.625);

  const applyFrame = (frame: Frame) => {
    const eng = engineRef.current;
    if (!eng || frame.header.codec !== "connectome_v1") return;
    if (!eng.fireTimeArr || !eng.fireTimeAttr) {
      eng.pendingFrame = frame;
      return;
    }
    const h = frame.header as unknown as ConnectomeFrameHeader;
    const fired = new Int32Array(frame.bits.buffer, frame.bits.byteOffset, Math.floor(frame.bits.byteLength / 4));
    const arr = eng.fireTimeArr;
    const nowSec = (performance.now() - startTimeRef.current) * 0.001;
    // In replay mode, clear prior spikes so scrubbed ticks display exact spikes.
    if (sourceRef.current === "replay") {
      arr.fill(-999);
    }
    const n = arr.length;
    if (fired.length > 0 || sourceRef.current === "replay") {
      let minIdx = n;
      let maxIdx = -1;
      for (let i = 0; i < fired.length; i++) {
        const idx = fired[i];
        if (idx >= 0 && idx < n) {
          arr[idx] = nowSec;
          if (idx < minIdx) minIdx = idx;
          if (idx > maxIdx) maxIdx = idx;
        }
      }
      if (sourceRef.current === "replay") {
        eng.fireTimeAttr.clearUpdateRanges();
        eng.fireTimeAttr.needsUpdate = true;
      } else if (maxIdx >= minIdx) {
        eng.fireTimeAttr.clearUpdateRanges();
        eng.fireTimeAttr.addUpdateRange(minIdx, maxIdx - minIdx + 1);
        eng.fireTimeAttr.needsUpdate = true;
      }
    }
    const nowMs = performance.now();
    if (nowMs - lastCountUiRef.current >= 200) {
      lastCountUiRef.current = nowMs;
      setActiveCount(h.active_neurons_tick ?? fired.length);
    }
    framesRef.current += 1;
  };

  useImperativeHandle(ref, () => ({ draw: applyFrame }), []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      });
    } catch {
      setStatus("webgl_unavailable");
      return;
    }

    const pr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pr);
    renderer.setSize(width, height);
    renderer.domElement.dataset.testid = testId;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050811);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 50);
    camera.position.set(0, 0.35, 2.65);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = autoRotateRef.current;
    controls.autoRotateSpeed = 0.6;
    controls.target.set(0, 0, 0);
    controls.update();

    const applyColors = (mode: "role" | "region") => {
      const eng = engineRef.current;
      if (!eng?.geom || !eng.colorAttr) return;
      const n = eng.geom.header.num_cells;
      const arr = eng.colorAttr.array as Float32Array;
      for (let i = 0; i < n; i++) {
        const rgb = mode === "region" ? regionColor(eng.geom.regionIdx[i]) : ROLE_COLORS[eng.geom.roles[i] % ROLE_COLORS.length];
        arr[3 * i] = rgb[0];
        arr[3 * i + 1] = rgb[1];
        arr[3 * i + 2] = rgb[2];
      }
      eng.colorAttr.needsUpdate = true;
    };

    const eng = {
      renderer,
      scene,
      camera,
      controls,
      geom: null as ConnectomeGeometry | null,
      points: null as THREE.Points | null,
      fireTimeAttr: null as THREE.BufferAttribute | null,
      colorAttr: null as THREE.BufferAttribute | null,
      fireTimeArr: null as Float32Array | null,
      timeUniform: null as { value: number } | null,
      decayUniform: null as { value: number } | null,
      pendingFrame: null as Frame | null,
      applyColors,
    };
    engineRef.current = eng;

    let cancelled = false;
    setStatus("loading");
    fetchGeometryCached(datasetId)
      .then((g) => {
        if (cancelled) return;
        const n = g.header.num_cells;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(g.positions, 3));
        const colors = new Float32Array(n * 3);
        const colorAttr = new THREE.BufferAttribute(colors, 3);
        geo.setAttribute("aColor", colorAttr);
        const fireTimeArr = new Float32Array(n);
        fireTimeArr.fill(-999);
        const fireTimeAttr = new THREE.BufferAttribute(fireTimeArr, 1);
        fireTimeAttr.setUsage(THREE.DynamicDrawUsage);
        geo.setAttribute("aFireTime", fireTimeAttr);

        // Larger base points for tiny synthetic circuits, crisp micro-points for 138k neurons.
        const baseSize = n <= 64 ? 22.0 : n <= 2000 ? 7.0 : 3.2;
        const timeUniform = { value: 0 };
        const decayUniform = { value: 5.5 };
        const mat = new THREE.ShaderMaterial({
          vertexShader: VERTEX_SHADER,
          fragmentShader: FRAGMENT_SHADER,
          uniforms: {
            uBaseSize: { value: baseSize },
            uPixelRatio: { value: pr },
            uTime: timeUniform,
            uDecayRate: decayUniform,
          },
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });

        const points = new THREE.Points(geo, mat);
        scene.add(points);
        eng.geom = g;
        eng.points = points;
        eng.fireTimeAttr = fireTimeAttr;
        eng.colorAttr = colorAttr;
        eng.fireTimeArr = fireTimeArr;
        eng.timeUniform = timeUniform;
        eng.decayUniform = decayUniform;
        applyColors(colorModeRef.current);
        setCellCount(n);
        setStatus("ready");
        if (eng.pendingFrame) {
          const pending = eng.pendingFrame;
          eng.pendingFrame = null;
          applyFrame(pending);
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });

    let raf = 0;
    const loop = () => {
      controls.autoRotate = autoRotateRef.current;
      controls.update();
      // Pure GPU shader decay: advance one float uniform (zero 138k CPU loop, zero PCIe upload!).
      if (eng.timeUniform && eng.decayUniform) {
        eng.timeUniform.value = (performance.now() - startTimeRef.current) * 0.001;
        eng.decayUniform.value = sourceRef.current === "replay" ? 0.0 : 5.5;
      }
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    const timer = window.setInterval(() => {
      onDisplayFps?.(framesRef.current);
      framesRef.current = 0;
    }, 1000);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      window.clearInterval(timer);
      if (eng.points) {
        eng.points.geometry.dispose();
        (eng.points.material as THREE.Material).dispose();
      }
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === host) host.removeChild(renderer.domElement);
      engineRef.current = null;
    };
  }, [datasetId, width, height, testId, onDisplayFps]);

  useEffect(() => {
    engineRef.current?.applyColors(colorMode);
  }, [colorMode]);

  return (
    <div className="field3d-wrap" data-testid="brain3d-wrap">
      <div ref={hostRef} className="field3d brain3d" style={{ width, height, position: "relative" }}>
        {status !== "ready" && (
          <div className="muted small" style={{ position: "absolute", top: 10, left: 12 }}>
            {status === "loading" ? t("common.loading") : t("connectome.brain3d.unavailable")}
          </div>
        )}
      </div>
      <div className="field3d-bar small">
        <span className="mono">
          {datasetId} · {cellCount.toLocaleString()} {t("connectome.cells")} · {t("connectome.active_tick")}: {activeCount}
        </span>
        <label className="inline">
          {t("connectome.brain3d.color")}
          <select value={colorMode} onChange={(e) => setColorMode(e.target.value as "role" | "region")} data-testid="brain3d-color">
            <option value="role">{t("connectome.brain3d.color_role")}</option>
            <option value="region">{t("connectome.brain3d.color_region")}</option>
          </select>
        </label>
        <label className="inline">
          <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} data-testid="brain3d-autorotate" />
          {t("connectome.brain3d.autorotate")}
        </label>
        <span className="muted">{t("connectome.brain3d.legend")}</span>
      </div>
    </div>
  );
});
