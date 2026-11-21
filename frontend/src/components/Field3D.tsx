// GPU presentation views for continuous fields (codec field_u8), drawn with three.js/WebGL2.
//   terrain   — the field as a lit landscape: height and colour = value (multi-channel fields:
//               height = brightest channel, colour = channels as RGB); bright parts glow.
//   spacetime — the last received frames stacked upwards into a translucent volume (newest on
//               top), rendered by ray marching: moving patterns leave tubes, splits leave forks.
// These are presentation views only. Heights, glow and opacity exaggerate on purpose; the
// 2D view remains the measurement view. Slices are received preview frames (which may be
// dropped under load), so their tick spacing is not constant; the tick range is shown.
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { Frame } from "../lib/api";
import { useT } from "../i18n";

export type Field3DMode = "terrain" | "spacetime";
export type Palette = "viridis" | "ember";

export interface Field3DHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  mode: Field3DMode;
  palette?: Palette;
  onDisplayFps: (fps: number) => void;
  testId?: string;
  width: number;
}

const PALETTES: Record<Palette, Array<[number, [number, number, number]]>> = {
  viridis: [
    [0, [10, 10, 24]],
    [0.25, [59, 40, 120]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1, [253, 231, 37]],
  ],
  ember: [
    [0, [8, 6, 10]],
    [0.3, [110, 30, 20]],
    [0.6, [230, 110, 30]],
    [0.85, [255, 200, 90]],
    [1, [255, 250, 220]],
  ],
};

function lutTexture(p: Palette): THREE.DataTexture {
  const data = new Uint8Array(256 * 4);
  const stops = PALETTES[p];
  const c = new THREE.Color();
  for (let v = 0; v < 256; v++) {
    const x = v / 255;
    let i = 0;
    while (i < stops.length - 2 && x > stops[i + 1][0]) i++;
    const [x0, c0] = stops[i];
    const [x1, c1] = stops[i + 1];
    const f = (x - x0) / (x1 - x0);
    c.setRGB(...(c0.map((a, k) => (a + f * (c1[k] - a)) / 255) as [number, number, number]), THREE.SRGBColorSpace);
    data.set([c.r * 255, c.g * 255, c.b * 255, 255], v * 4); // stored linear
  }
  const t = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

const TERRAIN_VS = /* glsl */ `
uniform sampler2D uField;
uniform vec2 uTexel;
uniform float uHeight;
uniform int uMulti;
varying vec2 vUv;
varying float vValue;
varying vec3 vNormal2;
float valueAt(vec2 uv) {
  vec4 s = texture2D(uField, vec2(uv.x, 1.0 - uv.y));
  // Colour mixes: between the brightest channel and the mean, so overlapping channels keep
  // their shape instead of forming a flat plateau.
  return uMulti == 1 ? 0.5 * (max(s.r, max(s.g, s.b)) + (s.r + s.g + s.b) / 3.0) : s.r;
}
void main() {
  vUv = uv;
  float v = valueAt(uv);
  vValue = v;
  float hl = valueAt(uv - vec2(uTexel.x, 0.0)), hr = valueAt(uv + vec2(uTexel.x, 0.0));
  float hd = valueAt(uv - vec2(0.0, uTexel.y)), hu = valueAt(uv + vec2(0.0, uTexel.y));
  vec3 n = normalize(vec3((hl - hr) * uHeight / (2.0 * uTexel.x), (hd - hu) * uHeight / (2.0 * uTexel.y), 1.0));
  vNormal2 = normalize(normalMatrix * n);
  vec3 p = position + vec3(0.0, 0.0, v * uHeight);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

const TERRAIN_FS = /* glsl */ `
uniform sampler2D uField;
uniform sampler2D uLut;
uniform int uMulti;
varying vec2 vUv;
varying float vValue;
varying vec3 vNormal2;
void main() {
  vec3 base;
  if (uMulti == 1) {
    vec3 s = texture2D(uField, vec2(vUv.x, 1.0 - vUv.y)).rgb;
    base = pow(s, vec3(2.2)) + vec3(0.015);
  } else {
    base = texture2D(uLut, vec2(vValue, 0.5)).rgb;
  }
  vec3 light = normalize(vec3(0.4, 0.5, 0.75));
  float diff = max(dot(normalize(vNormal2), normalize((viewMatrix * vec4(light, 0.0)).xyz)), 0.0);
  // Bright parts glow (less for colour mixes, which would otherwise wash out to white).
  vec3 col = base * (0.35 + 0.85 * diff) + base * vValue * vValue * (uMulti == 1 ? 0.25 : 0.7);
  // Faint reference grid (16 x 16 cells) so flat areas still show depth.
  vec2 g = abs(fract(vUv * 16.0 - 0.5) - 0.5) / fwidth(vUv * 16.0);
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  col += vec3(0.05, 0.07, 0.11) * line * (1.0 - smoothstep(0.05, 0.3, vValue)); // only on low ground
  gl_FragColor = vec4(col, 1.0);
}`;

const VOLUME_VS = /* glsl */ `
uniform vec3 uCamPos;
varying vec3 vOrigin;
varying vec3 vDir;
void main() {
  vOrigin = uCamPos;
  vDir = position - uCamPos;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const VOLUME_FS = /* glsl */ `
uniform highp sampler3D uVol;
uniform sampler2D uLut;
uniform int uMulti;
uniform float uDensity;
uniform float uFilled; // fraction of the stack that holds frames (fills up from the top)
varying vec3 vOrigin;
varying vec3 vDir;
vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(-0.5) - o) * inv, t1 = (vec3(0.5) - o) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
void main() {
  vec3 d = normalize(vDir);
  vec2 b = hitBox(vOrigin, d);
  if (b.x > b.y) discard;
  b.x = max(b.x, 0.0);
  const int STEPS = 220;
  float dt = (b.y - b.x) / float(STEPS);
  vec3 p = vOrigin + (b.x + 0.5 * dt) * d;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < STEPS; i++) {
    vec3 uvw = p + 0.5;
    if (uvw.z >= 1.0 - uFilled) {
      vec4 s = texture(uVol, vec3(uvw.x, 1.0 - uvw.y, uvw.z));
      float v = uMulti == 1 ? max(s.r, max(s.g, s.b)) : s.r;
      // Multi-channel: the hue of the mix at full brightness; the opacity carries the amount.
      vec3 col = uMulti == 1 ? pow(s.rgb / max(v, 1e-3), vec3(2.2)) : texture2D(uLut, vec2(v, 0.5)).rgb;
      float a = 1.0 - exp(-uDensity * v * v * dt);
      // The newest slice (top) is brighter, older ones fade a little.
      float age = 0.55 + 0.45 * uvw.z;
      acc.rgb += (1.0 - acc.a) * a * col * (0.45 + 1.1 * v) * age;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.97) break;
    }
    p += d * dt;
  }
  gl_FragColor = acc;
}`;

const STACK = 96; // frames in the space-time volume
const MAX_SIDE = 128; // volume texels per side (downsampled)

export const Field3D = forwardRef<Field3DHandle, Props>(function Field3D({ mode, palette = "viridis", onDisplayFps, testId = "world-canvas", width }, ref) {
  const t = useT();
  const host = useRef<HTMLDivElement>(null);
  const frames = useRef(0);
  const [autoRotate, setAutoRotate] = useState(false);
  const [heightScale, setHeightScale] = useState(0.15);
  const [every, setEvery] = useState(1); // stack one of every `every` received frames
  const everyRef = useRef(1);
  everyRef.current = every;
  const [ticks, setTicks] = useState<[number, number] | null>(null);
  const api = useRef<{ draw: (f: Frame) => void; setRotate: (on: boolean) => void; setHeight: (h: number) => void } | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.dataset.testid = testId;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070c);
    const camera = new THREE.PerspectiveCamera(40, 1.5, 0.01, 50);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = 0.8;
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), mode === "terrain" ? 0.45 : 0.7, 0.45, mode === "terrain" ? 0.6 : 0.4);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
    const lut = lutTexture(palette);

    let fieldTex: THREE.DataTexture | null = null;
    let terrain: THREE.Mesh | null = null;
    let volume: THREE.Mesh | null = null;
    let volTex: THREE.Data3DTexture | null = null;
    let signature = "";
    const slices: Array<{ tick: number; data: Uint8Array }> = [];
    let received = 0;
    let aspect = 1;

    const resize = () => {
      const w = el.clientWidth || width;
      const h = el.clientHeight || Math.round(w * 0.62);
      renderer.setSize(w, h);
      composer.setSize(w, h);
      bloom.resolution.set(w / 2, h / 2);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    const placeCamera = () => {
      if (mode === "terrain") camera.position.set(0, -1.25 * Math.max(1, aspect), 0.95);
      else camera.position.set(1.15 * Math.max(1, aspect), -1.35, 0.85);
      controls.target.set(0, 0, mode === "terrain" ? 0.05 : 0);
      controls.update();
    };

    const setup = (w: number, h: number, channels: number) => {
      for (const m of [terrain, volume]) if (m) scene.remove(m);
      scene.children.filter((c) => c.userData.frame).forEach((c) => scene.remove(c));
      aspect = w / h;
      const multi = channels > 1 ? 1 : 0;
      if (mode === "terrain") {
        fieldTex = new THREE.DataTexture(new Uint8Array(w * h * (multi ? 4 : 1)), w, h, multi ? THREE.RGBAFormat : THREE.RedFormat);
        fieldTex.unpackAlignment = 1;
        fieldTex.magFilter = fieldTex.minFilter = THREE.LinearFilter;
        fieldTex.wrapS = fieldTex.wrapT = THREE.RepeatWrapping; // the worlds are tori
        const seg = (n: number) => Math.min(n - 1, 383);
        const geo = new THREE.PlaneGeometry(aspect, 1, seg(w), seg(h));
        terrain = new THREE.Mesh(
          geo,
          new THREE.ShaderMaterial({
            vertexShader: TERRAIN_VS,
            fragmentShader: TERRAIN_FS,
            uniforms: {
              uField: { value: fieldTex },
              uLut: { value: lut },
              uTexel: { value: new THREE.Vector2(1 / w, 1 / h) },
              uHeight: { value: heightScale },
              uMulti: { value: multi },
            },
          }),
        );
        scene.add(terrain);
      } else {
        const f = Math.max(1, Math.ceil(Math.max(w, h) / MAX_SIDE));
        const vw = Math.ceil(w / f);
        const vh = Math.ceil(h / f);
        volTex = new THREE.Data3DTexture(new Uint8Array(vw * vh * STACK * (multi ? 4 : 1)), vw, vh, STACK);
        volTex.format = multi ? THREE.RGBAFormat : THREE.RedFormat;
        volTex.unpackAlignment = 1;
        volTex.magFilter = volTex.minFilter = THREE.LinearFilter;
        volTex.userData = { f, vw, vh, multi };
        const geo = new THREE.BoxGeometry(1, 1, 1);
        volume = new THREE.Mesh(
          geo,
          new THREE.ShaderMaterial({
            vertexShader: VOLUME_VS,
            fragmentShader: VOLUME_FS,
            uniforms: {
              uVol: { value: volTex },
              uLut: { value: lut },
              uMulti: { value: multi },
              uDensity: { value: 60 },
              uFilled: { value: 0 },
              uCamPos: { value: new THREE.Vector3() },
            },
            transparent: true,
            side: THREE.BackSide,
            depthWrite: false,
          }),
        );
        volume.scale.set(aspect, 1, 1.1);
        scene.add(volume);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x3a4660 }));
        edges.scale.copy(volume.scale);
        edges.userData.frame = true;
        scene.add(edges);
        slices.length = 0;
      }
      placeCamera();
    };

    // Box-average downsampling of one frame into a volume slice.
    const toSlice = (bytes: Uint8Array, w: number, h: number, c: number): Uint8Array => {
      const { f, vw, vh, multi } = volTex!.userData as { f: number; vw: number; vh: number; multi: number };
      const out = new Uint8Array(vw * vh * (multi ? 4 : 1));
      for (let y = 0; y < vh; y++) {
        for (let x = 0; x < vw; x++) {
          const acc = [0, 0, 0];
          let n = 0;
          for (let dy = 0; dy < f && y * f + dy < h; dy++) {
            for (let dx = 0; dx < f && x * f + dx < w; dx++) {
              const i = (y * f + dy) * w + (x * f + dx);
              for (let k = 0; k < Math.min(3, c); k++) acc[k] += bytes[i * c + k];
              n++;
            }
          }
          const o = y * vw + x;
          if (multi) {
            out[o * 4] = acc[0] / n;
            out[o * 4 + 1] = acc[1] / n;
            out[o * 4 + 2] = acc[2] / n;
            out[o * 4 + 3] = 255;
          } else out[o] = acc[0] / n;
        }
      }
      return out;
    };

    api.current = {
      draw: (frame: Frame) => {
        if (frame.header.codec !== "field_u8") return;
        const w = frame.header.width ?? 0;
        const h = frame.header.height ?? 0;
        const c = (frame.header as { channels?: number }).channels ?? 1;
        const sig = `${w}x${h}x${c}`;
        el.dataset.frame = sig;
        if (sig !== signature) {
          setup(w, h, c);
          signature = sig;
        }
        const bytes = frame.bits;
        if (mode === "terrain" && fieldTex) {
          const data = fieldTex.image.data as Uint8Array;
          if (c === 1) data.set(bytes.subarray(0, w * h));
          else
            for (let i = 0; i < w * h; i++) {
              for (let k = 0; k < 3; k++) data[i * 4 + k] = k < c ? bytes[i * c + k] : 0;
              data[i * 4 + 3] = 255;
            }
          fieldTex.needsUpdate = true;
        } else if (volTex && volume) {
          received += 1;
          if ((received - 1) % everyRef.current !== 0 && slices.length > 0) {
            frames.current += 1;
            return;
          }
          slices.push({ tick: frame.header.tick, data: toSlice(bytes, w, h, c) });
          if (slices.length > STACK) slices.shift();
          // Newest slice at the top (z = STACK - 1), older ones below.
          const data = volTex.image.data as Uint8Array;
          const per = slices[0].data.length;
          data.fill(0);
          slices.forEach((s, i) => data.set(s.data, (STACK - slices.length + i) * per));
          volTex.needsUpdate = true;
          (volume.material as THREE.ShaderMaterial).uniforms.uFilled.value = slices.length / STACK;
          el.dataset.slices = String(slices.length);
          if (frames.current % 5 === 0) setTicks([slices[0].tick, slices[slices.length - 1].tick]);
        }
        frames.current += 1;
      },
      setRotate: (on) => {
        controls.autoRotate = on;
      },
      setHeight: (hgt) => {
        if (terrain) (terrain.material as THREE.ShaderMaterial).uniforms.uHeight.value = hgt;
      },
    };

    if (early.current) {
      api.current.draw(early.current);
      early.current = null;
    }
    let raf = 0;
    const loop = () => {
      controls.update();
      if (volume) {
        volume.updateMatrixWorld();
        const local = volume.worldToLocal(camera.position.clone());
        (volume.material as THREE.ShaderMaterial).uniforms.uCamPos.value.copy(local);
      }
      composer.render();
      raf = requestAnimationFrame(loop);
    };
    loop();
    const timer = window.setInterval(() => {
      onDisplayFps(frames.current);
      frames.current = 0;
    }, 1000);
    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(timer);
      observer.disconnect();
      controls.dispose();
      composer.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      api.current = null;
    };
    // heightScale is applied through setHeight; re-creating the scene for it is not needed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, palette, onDisplayFps, testId, width]);

  useEffect(() => {
    api.current?.setRotate(autoRotate);
  }, [autoRotate]);
  useEffect(() => {
    api.current?.setHeight(heightScale);
  }, [heightScale]);
  // Frames that arrive before the scene exists (e.g. the last frame handed over right after
  // mounting) are kept and drawn once it is ready.
  const early = useRef<Frame | null>(null);
  useImperativeHandle(ref, () => ({
    draw: (f: Frame) => {
      if (api.current) api.current.draw(f);
      else early.current = f;
    },
  }), []);

  return (
    <div className="field3d-wrap">
      <div ref={host} className="field3d" style={{ width: "100%", maxWidth: width }} data-testid={`${testId}-3d`} />
      <div className="field3d-bar small">
        <span className="muted">{t(mode === "terrain" ? "view3d.terrain.note" : "view3d.spacetime.note")}</span>
        {mode === "spacetime" && ticks && (
          <span className="mono" data-testid="spacetime-ticks">
            {t("view3d.spacetime.range")} t={ticks[0]}…{ticks[1]}
          </span>
        )}
        {mode === "spacetime" && (
          <label className="inline">
            {t("view3d.every")}
            <select value={every} onChange={(e) => setEvery(Number(e.target.value))} data-testid="spacetime-every">
              {[1, 2, 4, 8, 16].map((k) => (
                <option key={k} value={k}>
                  {k === 1 ? t("view3d.every.all") : t("view3d.every.n").replace("{n}", String(k))}
                </option>
              ))}
            </select>
          </label>
        )}
        {mode === "terrain" && (
          <label className="inline">
            {t("view3d.height")}
            <input type="range" min={0.05} max={0.8} step={0.05} value={heightScale} onChange={(e) => setHeightScale(Number(e.target.value))} />
          </label>
        )}
        <label className="inline">
          <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} data-testid="view3d-rotate" />
          {t("view3d.rotate")}
        </label>
      </div>
    </div>
  );
});
