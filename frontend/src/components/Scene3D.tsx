// 3D view of embodied creatures (codec scene3d_f32): MuJoCo geoms as three.js meshes, z up.
// The header lists the geoms (type, size, rgba); the payload holds per geom position (3) and
// rotation matrix (9) as float32. The camera orbits (mouse) and follows the torso.
// Extras for readability: a rough terrain (fetched once, coloured by height, with a 1 m grid
// draped over it so the bumps show as bent lines), feet turn green
// while they touch the ground, and a line shows the torso's recent path.
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api, type Frame, type TerrainInfo } from "../lib/api";

export interface Scene3DHandle {
  draw: (frame: Frame) => void;
}

interface GeomInfo {
  name?: string;
  type: string;
  size: number[];
  rgba: number[];
}

interface Props {
  onDisplayFps: (fps: number) => void;
  onClick?: () => void;
  testId?: string;
  width?: number;
  height?: number;
  runId?: string;
}

const FOOT = new THREE.Color(0.2, 0.38, 0.75);
const FOOT_DOWN = new THREE.Color(0.15, 0.75, 0.3);

// Height colour ramp (low = sand, high = olive-brown), readable on the light background.
function terrainColour(v: number, out: THREE.Color): THREE.Color {
  const lo = new THREE.Color(0.93, 0.89, 0.78);
  const hi = new THREE.Color(0.48, 0.42, 0.28);
  return out.copy(lo).lerp(hi, Math.max(0, Math.min(1, v)));
}

export interface TerrainGrid {
  n: number;
  extent: number;
  amplitude: number;
  heights: Float32Array; // metres, row = y (from -extent/2), column = x
}

export function decodeTerrain(t: TerrainInfo): TerrainGrid {
  const raw = Uint8Array.from(atob(t.heights_u16_b64), (c) => c.charCodeAt(0));
  const u16 = new Uint16Array(raw.buffer);
  const heights = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) heights[i] = (u16[i] / 65535) * t.amplitude;
  return { n: t.n, extent: t.extent, amplitude: t.amplitude, heights };
}

export function groundAt(g: TerrainGrid | null, x: number, y: number): number {
  if (!g) return 0;
  const cell = g.extent / (g.n - 1);
  const j = Math.round((x + g.extent / 2) / cell);
  const i = Math.round((y + g.extent / 2) / cell);
  if (i < 0 || j < 0 || i >= g.n || j >= g.n) return 0;
  return g.heights[i * g.n + j];
}

function terrainMesh(g: TerrainGrid): THREE.Mesh {
  const geo = new THREE.PlaneGeometry(g.extent, g.extent, g.n - 1, g.n - 1);
  // PlaneGeometry vertices run row by row from +y (top) to -y; our rows start at -y.
  const pos = geo.attributes.position;
  const colours = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let k = 0; k < pos.count; k++) {
    const row = g.n - 1 - Math.floor(k / g.n);
    const col = k % g.n;
    const h = g.heights[row * g.n + col];
    pos.setZ(k, h);
    terrainColour(h / Math.max(g.amplitude, 1e-9), c).toArray(colours, 3 * k);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 }));
  mesh.receiveShadow = true;
  mesh.position.z = 0.001; // above the flat floor plane
  // A 1 m grid following the surface: straight lines on flat ground, bent lines over bumps.
  const pts: number[] = [];
  const cell = g.extent / (g.n - 1);
  const half = g.extent / 2;
  for (let line = -half; line <= half + 1e-9; line += 1) {
    for (let u = -half; u < half - 1e-9; u += cell) {
      for (const [a, b] of [[u, u + cell]]) {
        pts.push(line, a, groundAt(g, line, a) + 0.004, line, b, groundAt(g, line, b) + 0.004); // along y
        pts.push(a, line, groundAt(g, a, line) + 0.004, b, line, groundAt(g, b, line) + 0.004); // along x
      }
    }
  }
  const gridGeo = new THREE.BufferGeometry();
  gridGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  mesh.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({ color: 0x6b5d3e, transparent: true, opacity: 0.45 })));
  return mesh;
}

function meshFor(g: GeomInfo): THREE.Object3D {
  const colour = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
  const mat = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.6, metalness: 0.05 });
  let geo: THREE.BufferGeometry;
  if (g.type === "hfield") return new THREE.Group(); // drawn from the fetched terrain instead
  if (g.type === "plane") {
    const group = new THREE.Group();
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: colour, roughness: 0.95 }));
    floor.receiveShadow = true;
    const grid = new THREE.GridHelper(200, 200, 0x9aa3b2, 0xc4c9d2);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in xz; MuJoCo's floor is xy
    grid.position.z = 0.002;
    group.add(floor, grid);
    return group;
  }
  if (g.type === "sphere") geo = new THREE.SphereGeometry(g.size[0], 32, 16);
  else if (g.type === "capsule") {
    geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 8, 16);
    geo.rotateX(Math.PI / 2); // three's capsule is along y, MuJoCo's along local z
  } else if (g.type === "cylinder") {
    geo = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 24);
    geo.rotateX(Math.PI / 2);
  } else geo = new THREE.BoxGeometry(2 * g.size[0], 2 * (g.size[1] || g.size[0]), 2 * (g.size[2] || g.size[0]));
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  return mesh;
}

export const Scene3D = forwardRef<Scene3DHandle, Props>(function Scene3D({ onDisplayFps, onClick, testId = "world-canvas", width = 720, height = 520, runId }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const three = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    meshes: THREE.Object3D[];
    prevPos: THREE.Vector3[];
    prevQuat: THREE.Quaternion[];
    targetPos: THREE.Vector3[];
    targetQuat: THREE.Quaternion[];
    interpStartMs: number;
    frameIntervalMs: number;
    lastFrameArrivalMs: number;
    signature: string;
    lastTorso: THREE.Vector3 | null;
    trail: THREE.Line;
    trailBuf: Float32Array;
  } | null>(null);
  const scratchMat4 = useRef(new THREE.Matrix4());
  const terrain = useRef<TerrainGrid | null>(null);
  const frames = useRef(0);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.domElement.dataset.testid = testId;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeef1f5);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, -4, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(sun, sun.target);
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 500);
    camera.up.set(0, 0, 1);
    camera.position.set(2.5, -3.5, 2.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.3);
    controls.update();
    const trailBuf = new Float32Array(512 * 3);
    const trailGeo = new THREE.BufferGeometry();
    const trailAttr = new THREE.BufferAttribute(trailBuf, 3);
    trailAttr.setUsage(THREE.DynamicDrawUsage);
    trailGeo.setAttribute("position", trailAttr);
    trailGeo.setDrawRange(0, 0);
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xd9480f }));
    trail.frustumCulled = false;
    scene.add(trail);
    three.current = {
      renderer,
      scene,
      camera,
      controls,
      meshes: [],
      prevPos: [],
      prevQuat: [],
      targetPos: [],
      targetQuat: [],
      interpStartMs: performance.now(),
      frameIntervalMs: 18,
      lastFrameArrivalMs: 0,
      signature: "",
      lastTorso: null,
      trail,
      trailBuf,
    };
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const st = three.current;
      if (st && st.meshes.length > 0) {
        const span = Math.max(14, Math.min(120, st.frameIntervalMs * 1.05));
        const alpha = Math.max(0, Math.min(1, (performance.now() - st.interpStartMs) / span));
        for (let i = 0; i < st.meshes.length; i++) {
          st.meshes[i].position.lerpVectors(st.prevPos[i], st.targetPos[i], alpha);
          st.meshes[i].quaternion.slerpQuaternions(st.prevQuat[i], st.targetQuat[i], alpha);
        }
      }
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    const timer = window.setInterval(() => {
      onDisplayFps(frames.current);
      frames.current = 0;
    }, 1000);
    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(timer);
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      three.current = null;
    };
  }, [width, height, onDisplayFps, testId]);

  // Static scene data (terrain) is fetched once per run, not sent with every frame.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    let mesh: THREE.Mesh | null = null;
    api
      .scene(runId)
      .then((r) => {
        const t = three.current;
        if (cancelled || !t || !r.terrain) return;
        terrain.current = decodeTerrain(r.terrain);
        mesh = terrainMesh(terrain.current);
        t.scene.add(mesh);
      })
      .catch(() => undefined); // the view still works without terrain
    return () => {
      cancelled = true;
      if (mesh) three.current?.scene.remove(mesh);
      terrain.current = null;
    };
  }, [runId, width, height]);

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => {
      const t = three.current;
      if (!t || frame.header.codec !== "scene3d_f32") return;
      const header = frame.header as unknown as {
        geoms: GeomInfo[];
        count: number;
        torso: number[];
        contacts?: boolean[];
        trail?: number[][];
      };
      const nowMs = performance.now();
      if (t.lastFrameArrivalMs > 0) {
        const dtArrival = nowMs - t.lastFrameArrivalMs;
        if (dtArrival >= 5 && dtArrival <= 250) {
          t.frameIntervalMs = t.frameIntervalMs * 0.78 + dtArrival * 0.22;
        }
      }
      t.lastFrameArrivalMs = nowMs;
      t.interpStartMs = nowMs;

      const signature = JSON.stringify(header.geoms);
      let snapImmediate = false;
      if (signature !== t.signature) {
        for (const m of t.meshes) t.scene.remove(m);
        t.prevPos = [];
        t.prevQuat = [];
        t.targetPos = [];
        t.targetQuat = [];
        t.meshes = header.geoms.map((g) => {
          const m = meshFor(g);
          t.scene.add(m);
          t.prevPos.push(new THREE.Vector3());
          t.prevQuat.push(new THREE.Quaternion());
          t.targetPos.push(new THREE.Vector3());
          t.targetQuat.push(new THREE.Quaternion());
          return m;
        });
        t.signature = signature;
        snapImmediate = true;
      }
      // The payload may start at any byte offset inside the message; copy to an aligned buffer.
      const f = new Float32Array(frame.bits.slice().buffer);
      const mat4 = scratchMat4.current;
      header.geoms.forEach((g, i) => {
        const o = i * 12;
        const m = t.meshes[i];
        if (!snapImmediate) {
          t.prevPos[i].copy(m.position);
          t.prevQuat[i].copy(m.quaternion);
        }
        if (g.type === "plane") {
          t.targetPos[i].set(0, 0, 0);
          t.targetQuat[i].identity();
        } else {
          t.targetPos[i].set(f[o], f[o + 1], f[o + 2]);
          mat4.set(
            f[o + 3], f[o + 4], f[o + 5], 0,
            f[o + 6], f[o + 7], f[o + 8], 0,
            f[o + 9], f[o + 10], f[o + 11], 0,
            0, 0, 0, 1,
          );
          t.targetQuat[i].setFromRotationMatrix(mat4);
        }
        if (snapImmediate) {
          t.prevPos[i].copy(t.targetPos[i]);
          t.prevQuat[i].copy(t.targetQuat[i]);
          m.position.copy(t.targetPos[i]);
          m.quaternion.copy(t.targetQuat[i]);
        }
        const foot = g.name ? /^foot(\d+)$/.exec(g.name) : null;
        if (foot && m instanceof THREE.Mesh) {
          (m.material as THREE.MeshStandardMaterial).color.copy(header.contacts?.[Number(foot[1])] ? FOOT_DOWN : FOOT);
        }
      });
      if (header.trail && header.trail.length > 1) {
        const totalPts = Math.min(header.trail.length + 1, t.trailBuf.length / 3);
        const keepTrail = totalPts - 1;
        const startIdx = header.trail.length - keepTrail;
        for (let i = 0; i < keepTrail; i++) {
          const [tx, ty] = header.trail[startIdx + i];
          t.trailBuf[i * 3] = tx;
          t.trailBuf[i * 3 + 1] = ty;
          t.trailBuf[i * 3 + 2] = groundAt(terrain.current, tx, ty) + 0.02;
        }
        t.trailBuf[keepTrail * 3] = header.torso[0];
        t.trailBuf[keepTrail * 3 + 1] = header.torso[1];
        t.trailBuf[keepTrail * 3 + 2] = groundAt(terrain.current, header.torso[0], header.torso[1]) + 0.02;
        (t.trail.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
        t.trail.geometry.setDrawRange(0, totalPts);
      } else {
        t.trail.geometry.setDrawRange(0, 0);
      }
      // Follow the torso: shift camera and target by the torso's planar motion.
      const torso = new THREE.Vector3(header.torso[0], header.torso[1], 0);
      if (t.lastTorso) {
        const delta = torso.clone().sub(t.lastTorso);
        t.camera.position.add(delta);
        t.controls.target.add(delta);
      } else {
        t.controls.target.set(torso.x, torso.y, 0.3);
      }
      t.lastTorso = torso;
      frames.current += 1;
    },
  }));

  return <div ref={host} className="scene3d" onClick={onClick} style={{ width, height, maxWidth: "100%" }} />;
});
