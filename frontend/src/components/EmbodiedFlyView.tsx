// Phase F8 Level 1: Side-by-side 2-pane Embodied Fly view (3D MuJoCo Arena + 3D Glowing Brain)
// with live descending/motor command meters, timing breakdown, body-part circuit links, and
// live arena/brain interventions.
import { forwardRef, lazy, Suspense, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { Frame } from "../lib/api";
import { useT } from "../i18n";
import type { Brain3DHandle } from "./Brain3D";

const Brain3D = lazy(() => import("./Brain3D").then((m) => ({ default: m.Brain3D })));

// Cached CT-scanned Apache-2.0 NeuroMechFly v2 STL loader (gracefully falls back offline).
const STL_CACHE = new Map<string, Promise<THREE.BufferGeometry | null>>();
const STL_LOADER = new STLLoader();

function loadNmfStl(name: string): Promise<THREE.BufferGeometry | null> {
  let p = STL_CACHE.get(name);
  if (!p) {
    p = fetch(`/api/v1/connectome/neuromechfly/mesh/${name}`)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => {
        if (!buf) return null;
        const geo = STL_LOADER.parse(buf);
        geo.computeVertexNormals();
        return geo;
      })
      .catch(() => null);
    STL_CACHE.set(name, p);
  }
  return p;
}

// Convert NeuroMechFly rigging mm coordinates -> our cm MuJoCo fly frame (1 mm * 0.27 = 0.27 units).
const NMF_MM_TO_SIM = 0.27;
const NMF_STL_SCALE = 1000.0 * NMF_MM_TO_SIM; // STL vertices are in metres

function addStlPart(
  parent: THREE.Group,
  geo: THREE.BufferGeometry,
  mat: THREE.Material,
  posMm: [number, number, number],
  quatWxyz: [number, number, number, number],
  mirrorY = false,
): THREE.Mesh {
  const m = new THREE.Mesh(geo, mat);
  m.scale.set(NMF_STL_SCALE, mirrorY ? -NMF_STL_SCALE : NMF_STL_SCALE, NMF_STL_SCALE);
  m.position.set(posMm[0] * NMF_MM_TO_SIM, posMm[1] * NMF_MM_TO_SIM, posMm[2] * NMF_MM_TO_SIM);
  m.quaternion.set(quatWxyz[1], quatWxyz[2], quatWxyz[3], quatWxyz[0]);
  if (mirrorY) {
    const matDouble = mat.clone();
    matDouble.side = THREE.DoubleSide;
    m.material = matDouble;
  }
  m.castShadow = true;
  parent.add(m);
  return m;
}

async function buildCtThoraxGroup(): Promise<THREE.Group | null> {
  const names = [
    "c_thorax.stl",
    "c_abdomen12.stl",
    "c_abdomen3.stl",
    "c_abdomen4.stl",
    "c_abdomen5.stl",
    "c_abdomen6.stl",
    "l_wing.stl",
    "l_haltere.stl",
  ] as const;
  const geos = await Promise.all(names.map(loadNmfStl));
  if (!geos[0]) return null;

  // Verified world clearance & neck alignment on unrotated MuJoCo BODY thorax (+X forward, +Z up):
  // c_thorax front (+0.04mm) -> +0.160 relative to thorax (meets head back cleanly); lowest vertex z=+0.080.
  const root = new THREE.Group();
  const shift = new THREE.Group();
  shift.position.set(0.55 * NMF_MM_TO_SIM, 0, 0.20 * NMF_MM_TO_SIM);
  root.add(shift);

  const thoraxMat = new THREE.MeshPhysicalMaterial({
    color: 0x9c6628,
    roughness: 0.42,
    metalness: 0.08,
    clearcoat: 0.45,
    clearcoatRoughness: 0.35,
  });
  const abdGoldMat = new THREE.MeshPhysicalMaterial({
    color: 0xc9933c,
    roughness: 0.36,
    metalness: 0.06,
    clearcoat: 0.55,
  });
  const abdDarkMat = new THREE.MeshPhysicalMaterial({
    color: 0x261a10,
    roughness: 0.32,
    metalness: 0.1,
    clearcoat: 0.6,
  });
  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xdbeafe,
    roughness: 0.15,
    metalness: 0.05,
    transmission: 0.55,
    transparent: true,
    opacity: 0.46,
    side: THREE.DoubleSide,
  });

  addStlPart(shift, geos[0], thoraxMat, [0, 0, 0], [1, 0, 0, 0]);
  // Chained abdomen segments from rigging.yaml (alternating golden & melanic Drosophila tergites).
  const abdRig: Array<[number, [number, number, number], [number, number, number, number], THREE.Material]> = [
    [1, [-0.862, 0.0, 0.0], [0.991, 0.0, -0.131, 0.0], abdGoldMat],
    [2, [-0.326, 0.0, 0.04], [0.999, 0.0, -0.043, 0.0], abdDarkMat],
    [3, [-0.208, 0.0, 0.025], [0.999, 0.0, 0.043, 0.0], abdGoldMat],
    [4, [-0.174, 0.0, 0.009], [1.0, 0.0, 0.009, 0.0], abdDarkMat],
    [5, [-0.228, 0.0, 0.009], [1.0, 0.0, 0.009, 0.0], abdDarkMat],
  ];
  let parentChain: THREE.Group = shift;
  for (const [gi, pos, quat, mat] of abdRig) {
    const g = geos[gi];
    if (!g) break;
    const node = new THREE.Group();
    node.position.set(pos[0] * NMF_MM_TO_SIM, pos[1] * NMF_MM_TO_SIM, pos[2] * NMF_MM_TO_SIM);
    node.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
    parentChain.add(node);
    addStlPart(node, g, mat, [0, 0, 0], [1, 0, 0, 0]);
    parentChain = node;
  }
  // Left & mirrored right CT-scanned translucent wings (folded cleanly back over abdomen).
  if (geos[6]) {
    addStlPart(shift, geos[6], wingMat, [-0.674, 0.34, 0.14], [0.649, 0.036, -0.065, 0.757], false);
    addStlPart(shift, geos[6], wingMat, [-0.674, -0.34, 0.14], [0.649, -0.036, -0.065, -0.757], true);
  }
  if (geos[7]) {
    addStlPart(shift, geos[7], abdGoldMat, [-0.806, 0.214, -0.149], [0.946, 0.124, 0.039, 0.298], false);
    addStlPart(shift, geos[7], abdGoldMat, [-0.806, -0.214, -0.149], [0.946, -0.124, 0.039, -0.298], true);
  }
  return root;
}

async function buildCtHeadGroup(): Promise<THREE.Group | null> {
  const names = ["c_head.stl", "l_eye.stl", "l_pedicel.stl", "l_funiculus.stl", "l_arista.stl"] as const;
  const geos = await Promise.all(names.map(loadNmfStl));
  if (!geos[0]) return null;

  const root = new THREE.Group();
  const shift = new THREE.Group();
  shift.position.set(-0.20 * NMF_MM_TO_SIM, 0, 0.02 * NMF_MM_TO_SIM);
  root.add(shift);

  const headMat = new THREE.MeshPhysicalMaterial({
    color: 0x8b5a25,
    roughness: 0.45,
    clearcoat: 0.35,
  });
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0xcc1d1d,
    emissive: 0x450a0a,
    emissiveIntensity: 0.35,
    roughness: 0.18,
    metalness: 0.12,
    clearcoat: 0.9,
    clearcoatRoughness: 0.12,
  });
  const antMat = new THREE.MeshStandardMaterial({ color: 0xdfb164, roughness: 0.5 });

  addStlPart(shift, geos[0], headMat, [0, 0, 0], [1, 0, 0, 0]);
  if (geos[1]) {
    addStlPart(shift, geos[1], eyeMat, [0.231, 0.0, 0.017], [1, 0, 0, 0], false);
    addStlPart(shift, geos[1], eyeMat, [0.231, 0.0, 0.017], [1, 0, 0, 0], true);
  }
  // Bilateral antennae chain (pedicel -> funiculus -> arista)
  for (const mirror of [false, true]) {
    const sy = mirror ? -1 : 1;
    if (geos[2]) {
      const ped = new THREE.Group();
      ped.position.set(0.429 * NMF_MM_TO_SIM, sy * 0.086 * NMF_MM_TO_SIM, 0.083 * NMF_MM_TO_SIM);
      ped.quaternion.set(sy * 0.2613, 0.0729, -sy * 0.0194, 0.9623);
      shift.add(ped);
      addStlPart(ped, geos[2], antMat, [0, 0, 0], [1, 0, 0, 0], mirror);
      if (geos[3]) {
        const fun = new THREE.Group();
        fun.position.set(0.051 * NMF_MM_TO_SIM, -sy * 0.000563 * NMF_MM_TO_SIM, -0.0441 * NMF_MM_TO_SIM);
        ped.add(fun);
        addStlPart(fun, geos[3], antMat, [0, 0, 0], [1, 0, 0, 0], mirror);
        if (geos[4]) {
          addStlPart(fun, geos[4], antMat, [0.0198, sy * 0.0539, -0.0683], [1, 0, 0, 0], mirror);
        }
      }
    }
  }
  return root;
}

async function buildCtProboscisGroup(): Promise<THREE.Group | null> {
  const [ros, hau] = await Promise.all([loadNmfStl("c_rostrum.stl"), loadNmfStl("c_haustellum.stl")]);
  if (!ros) return null;
  const root = new THREE.Group();
  const mat = new THREE.MeshPhysicalMaterial({ color: 0xd98a3c, roughness: 0.35, clearcoat: 0.5 });
  const shift = new THREE.Group();
  shift.position.set(-0.01, 0, 0.015);
  shift.scale.setScalar(0.72);
  root.add(shift);
  addStlPart(shift, ros, mat, [0, 0, 0], [1, 0, 0, 0]);
  if (hau) {
    addStlPart(shift, hau, mat, [-0.371, 0, -0.0196], [1, 0, 0, 0]);
  }
  return root;
}

export interface EmbodiedFlyViewHandle {
  draw: (frame: Frame) => void;
}

interface GeomInfo {
  name?: string;
  type: string;
  size: number[];
  rgba: number[];
}

interface FoodItem {
  x: number;
  y: number;
  radius: number;
  kind: "sugar" | "bitter";
}

interface BodyPose {
  pos: [number, number, number];
  mat: number[];
}

interface EmbodiedHeader {
  codec: "fly_embodied_v1";
  tick: number;
  dataset_id: string;
  wiring: string;
  num_cells: number;
  geom_count: number;
  geom_bytes: number;
  fired_count: number;
  geoms: GeomInfo[];
  body_poses?: {
    thorax?: BodyPose;
    head?: BodyPose;
    proboscis?: BodyPose;
  };
  torso: [number, number, number];
  yaw: number;
  arena_half_size: number;
  foods: FoodItem[];
  trail: Array<[number, number]>;
  poke_remaining_ms: number;
  commands: {
    forward: number;
    turn: number;
    backward: number;
    proboscis: number;
    grooming: number;
  };
  sensors: {
    sugar: number;
    bitter: number;
    antennal_touch: number;
    odor_left: number;
    odor_right: number;
    contacts: boolean[];
  };
  timing_ms: {
    brain: number;
    body: number;
    transfer: number;
    total: number;
    realtime_factor: number;
  };
  mn9_silenced: boolean;
  control_mode?: "level1_cpg" | "level2_vnc" | "level2_vnc_tuned";
  vnc_profile?: string | null;
  alif_enabled?: boolean;
  has_vnc_ports?: boolean;
  vnc_mn_mean_hz?: number;
  vnc_tripod_alternation?: number;
  proprio?: number[];
  leg_mn_hz?: Record<string, Record<string, number>> | null;
  has_stl_meshes?: boolean;
  body_asset_source: string;
  level_note: string;
  evolved_descendant?: {
    fly_id: number | null;
    generation: number | null;
    fly_distance: number;
    sugar_scale: number;
    bitter_scale: number;
    touch_scale: number;
    body_scale: number;
    leg_length: number;
    wing_span: number;
  } | null;
}

interface Props {
  datasetId: string;
  live: boolean;
  frameSource: string;
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onDisplayFps?: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

const TARSUS_AIR = new THREE.Color(0.88, 0.65, 0.28);
const TARSUS_DOWN = new THREE.Color(0.22, 0.85, 0.42);

function makeGeomMesh(g: GeomInfo, arenaHalfSize: number): THREE.Object3D {
  const col = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
  const transparent = g.rgba[3] < 0.99;
  const mat = new THREE.MeshStandardMaterial({
    color: col,
    roughness: transparent ? 0.25 : 0.55,
    metalness: 0.08,
    transparent,
    opacity: g.rgba[3],
  });
  if (g.type === "plane") {
    const size = Math.max(4.8, arenaHalfSize * 2.0);
    const divs = Math.max(16, Math.round(size * 2.5));
    const group = new THREE.Group();
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(size + 0.8, size + 0.8),
      new THREE.MeshStandardMaterial({ color: 0x1a2230, roughness: 0.92 }),
    );
    floor.receiveShadow = true;
    const grid = new THREE.GridHelper(size, divs, 0x3b485e, 0x263042);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    // Crisp arena boundary rim so the petri-dish perimeter is immediately legible.
    const rim = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-arenaHalfSize, -arenaHalfSize, 0.006),
        new THREE.Vector3(arenaHalfSize, -arenaHalfSize, 0.006),
        new THREE.Vector3(arenaHalfSize, arenaHalfSize, 0.006),
        new THREE.Vector3(-arenaHalfSize, arenaHalfSize, 0.006),
      ]),
      new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.65 }),
    );
    group.add(floor, grid, rim);
    return group;
  }
  let geo: THREE.BufferGeometry;
  if (g.type === "sphere") {
    geo = new THREE.SphereGeometry(g.size[0], 20, 14);
  } else if (g.type === "capsule") {
    geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 8, 14);
    geo.rotateX(Math.PI / 2);
  } else {
    geo = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
  }
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  return mesh;
}

function makeFoodObject(f: FoodItem): THREE.Group {
  const group = new THREE.Group();
  group.position.set(f.x, f.y, 0);
  if (f.kind === "sugar") {
    // Glossy honey-gold nectar droplet dome (compact & realistic vs fly head size) + satellite drops.
    const dropMat = new THREE.MeshPhysicalMaterial({
      color: 0xf59e0b,
      emissive: 0x78350f,
      emissiveIntensity: 0.35,
      roughness: 0.14,
      metalness: 0.05,
      clearcoat: 0.9,
      clearcoatRoughness: 0.1,
      transparent: true,
      opacity: 0.92,
    });
    const mainDrop = new THREE.Mesh(new THREE.SphereGeometry(0.14, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.58), dropMat);
    mainDrop.rotation.x = Math.PI / 2;
    mainDrop.scale.set(1.0, 1.0, 0.68);
    mainDrop.position.z = 0.004;
    group.add(mainDrop);

    // Two tiny glistening satellite nectar beads for instant "liquid food droplet" readability.
    for (const [dx, dy, r] of [
      [0.14, 0.08, 0.048],
      [-0.12, -0.09, 0.04],
    ] as const) {
      const bead = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.58), dropMat);
      bead.rotation.x = Math.PI / 2;
      bead.scale.set(1.0, 1.0, 0.72);
      bead.position.set(dx, dy, 0.004);
      group.add(bead);
    }

    // Subtle tasting-zone ring & soft golden odor plume mist on the floor.
    const tasteRing = new THREE.Mesh(
      new THREE.RingGeometry(Math.max(0.16, f.radius - 0.025), f.radius, 36),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.42, side: THREE.DoubleSide }),
    );
    tasteRing.position.z = 0.007;
    group.add(tasteRing);

    const plume = new THREE.Mesh(
      new THREE.RingGeometry(f.radius, f.radius + 1.05, 40),
      new THREE.MeshBasicMaterial({ color: 0xfde68a, transparent: true, opacity: 0.09, side: THREE.DoubleSide }),
    );
    plume.position.z = 0.005;
    group.add(plume);
  } else {
    // Bitter compound: sharp dark-violet crystal cluster (distinct shape & color from honey drop).
    const crystalMat = new THREE.MeshStandardMaterial({
      color: 0xa855f7,
      emissive: 0x4c1d95,
      emissiveIntensity: 0.45,
      roughness: 0.22,
      metalness: 0.18,
    });
    const center = new THREE.Mesh(new THREE.OctahedronGeometry(0.11, 0), crystalMat);
    center.scale.set(1.0, 1.0, 0.78);
    center.position.z = 0.065;
    group.add(center);

    for (const [dx, dy, s, rz] of [
      [0.11, 0.05, 0.065, 0.6],
      [-0.09, 0.09, 0.055, 1.4],
      [-0.06, -0.10, 0.06, 2.3],
    ] as const) {
      const shard = new THREE.Mesh(new THREE.OctahedronGeometry(s, 0), crystalMat);
      shard.rotation.z = rz;
      shard.position.set(dx, dy, s * 0.65);
      group.add(shard);
    }

    const tasteRing = new THREE.Mesh(
      new THREE.RingGeometry(Math.max(0.15, f.radius - 0.025), f.radius, 36),
      new THREE.MeshBasicMaterial({ color: 0xc084fc, transparent: true, opacity: 0.45, side: THREE.DoubleSide }),
    );
    tasteRing.position.z = 0.007;
    group.add(tasteRing);
  }
  return group;
}

type BodyLinkKey = "proboscis" | "antennae" | "legs" | null;

const BODY_LINKS: Record<Exclude<BodyLinkKey, null>, { labelJa: string; labelEn: string; detailJa: string; detailEn: string }> = {
  proboscis: {
    labelJa: "口（吻） ↔ MN9（CB0701）",
    labelEn: "Proboscis ↔ MN9 (CB0701)",
    detailJa: "甘味細胞（sugar_grn）の入力で MN9 が発火して口が伸び、苦味細胞（bitter_grn）で抑えられます。",
    detailEn: "Sweet taste (sugar_grn) excites MN9 to extend the proboscis; bitter taste (bitter_grn) suppresses it.",
  },
  antennae: {
    labelJa: "触角 ↔ DNg84 / ORN",
    labelEn: "Antennae ↔ DNg84 / ORN",
    detailJa: "左右の触角の匂い差で旋回（DNa01/02）し、触角への接触で毛づくろい細胞（DNg84）が発火します。",
    detailEn: "Bilateral odor differences steer DNa01/02; antennal touch triggers grooming neuron DNg84.",
  },
  legs: {
    labelJa: "6本の脚 ↔ DNp09 / DNa01-02 / MDN ＋ レベル1歩行器",
    labelEn: "Six legs ↔ DNp09 / DNa01-02 / MDN + Level 1 CPG",
    detailJa: "脳からの前進・旋回・後退指令を受け、手書きの3脚交代歩行器（CPG）とつまずき反射が脚を動かします。",
    detailEn: "Descending commands modulate a hand-crafted tripod CPG and stumble reflex (Level 1).",
  },
};

export const EmbodiedFlyView = forwardRef<EmbodiedFlyViewHandle, Props>(function EmbodiedFlyView(
  { datasetId, live, frameSource, onIntervene, onDisplayFps, maxPixels = 1040, testId = "world-canvas" },
  ref,
) {
  const t = useT();
  const arenaHostRef = useRef<HTMLDivElement>(null);
  const brainRef = useRef<Brain3DHandle | null>(null);
  const lastBrainFrame = useRef<Frame | null>(null);

  const [hud, setHud] = useState<EmbodiedHeader | null>(null);
  const [webglStatus, setWebglStatus] = useState<"ok" | "unavailable">("ok");
  const [gpuRenderer, setGpuRenderer] = useState<string>("");
  const [renderFps, setRenderFps] = useState<number>(0);
  const [selectedPart, setSelectedPart] = useState<BodyLinkKey>(null);

  const arenaRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    meshes: THREE.Object3D[];
    prevPos: THREE.Vector3[];
    prevQuat: THREE.Quaternion[];
    targetPos: THREE.Vector3[];
    targetQuat: THREE.Quaternion[];
    // Dedicated unrotated MuJoCo body mounts for CT-scanned STL meshes (+X anterior, +Z dorsal)
    ctMounts: Record<"thorax" | "head" | "proboscis", THREE.Group>;
    ctPrevPos: Record<"thorax" | "head" | "proboscis", THREE.Vector3>;
    ctPrevQuat: Record<"thorax" | "head" | "proboscis", THREE.Quaternion>;
    ctTargetPos: Record<"thorax" | "head" | "proboscis", THREE.Vector3>;
    ctTargetQuat: Record<"thorax" | "head" | "proboscis", THREE.Quaternion>;
    ctHasPose: boolean;
    interpStartMs: number;
    frameIntervalMs: number;
    lastFrameArrivalMs: number;
    targetTorso: THREE.Vector3 | null;
    tarsusIndices: number[];
    foodGroup: THREE.Group;
    trailLine: THREE.Line;
    trailBuf: Float32Array;
    geomSignature: string;
    foodSignature: string;
    followInitialized: boolean;
  } | null>(null);
  const framesRef = useRef(0);
  const rafFramesRef = useRef(0);
  const lastHudTsRef = useRef(0);
  const scratchMat4Ref = useRef(new THREE.Matrix4());

  const paneWidth = Math.max(320, Math.floor((maxPixels - 12) / 2));
  const paneHeight = Math.round(paneWidth * 0.68);

  const attachBrain = useCallback((h: Brain3DHandle | null) => {
    brainRef.current = h;
    if (h && lastBrainFrame.current) h.draw(lastBrainFrame.current);
  }, []);

  const drawFrame = useCallback((frame: Frame) => {
    if (frame.header.codec !== "fly_embodied_v1") return;
    const h = frame.header as unknown as EmbodiedHeader;
    // Throttle React HUD re-renders to ~10 Hz so 3D WebGL requestAnimationFrame never stutters.
    const nowMs = performance.now();
    if (nowMs - lastHudTsRef.current >= 95 || h.tick === 0) {
      lastHudTsRef.current = nowMs;
      setHud(h);
    }
    framesRef.current += 1;

    // Forward the neural spike payload to the right-hand Brain3D pane on the exact same tick!
    const geomBytes = h.geom_bytes ?? h.geom_count * 12 * 4;
    const firedSlice = frame.bits.subarray(geomBytes);
    const brainFrame: Frame = {
      header: {
        codec: "connectome_v1",
        tick: h.tick,
        num_cells: h.num_cells,
        active_neurons_tick: h.fired_count,
        fired_count: h.fired_count,
      } as unknown as Frame["header"],
      bits: firedSlice,
    };
    lastBrainFrame.current = brainFrame;
    brainRef.current?.draw(brainFrame);

    // Update snapshot targets for true constant-velocity interpolation in rAF.
    const eng = arenaRef.current;
    if (!eng) return;

    // Measure EWMA frame interval so snapshot interpolation adapts seamlessly to 30/60/120 FPS & any speed.
    if (eng.lastFrameArrivalMs > 0) {
      const dtArrival = nowMs - eng.lastFrameArrivalMs;
      if (dtArrival >= 5 && dtArrival <= 250) {
        eng.frameIntervalMs = eng.frameIntervalMs * 0.78 + dtArrival * 0.22;
      }
    }
    eng.lastFrameArrivalMs = nowMs;
    eng.interpStartMs = nowMs;

    const arenaHalf = h.arena_half_size ?? 6.0;
    const useCtBody = Boolean(h.has_stl_meshes && h.body_poses?.thorax && h.body_poses?.head);
    const sig = `${arenaHalf}|${useCtBody ? 1 : 0}|` + h.geoms.map((g) => `${g.name ?? ""}:${g.type}`).join("|");
    let snapImmediate = false;
    if (sig !== eng.geomSignature) {
      for (const m of eng.meshes) eng.scene.remove(m);
      for (const k of ["thorax", "head", "proboscis"] as const) {
        while (eng.ctMounts[k].children.length > 0) {
          eng.ctMounts[k].remove(eng.ctMounts[k].children[0]);
        }
      }
      eng.meshes = [];
      eng.prevPos = [];
      eng.prevQuat = [];
      eng.targetPos = [];
      eng.targetQuat = [];
      eng.tarsusIndices = [];
      eng.ctHasPose = false;
      const placeholderHide: THREE.Object3D[] = [];

      h.geoms.forEach((g, idx) => {
        const obj = makeGeomMesh(g, arenaHalf);
        const nm = g.name ?? "";
        eng.scene.add(obj);
        eng.meshes.push(obj);
        if (
          nm === "g_thorax" ||
          nm === "g_head" ||
          nm === "g_proboscis" ||
          nm.startsWith("g_abdomen") ||
          nm.startsWith("g_wing") ||
          nm.startsWith("g_eye") ||
          nm.startsWith("g_ant")
        ) {
          placeholderHide.push(obj);
        }
        eng.prevPos.push(new THREE.Vector3());
        eng.prevQuat.push(new THREE.Quaternion());
        eng.targetPos.push(new THREE.Vector3());
        eng.targetQuat.push(new THREE.Quaternion());
        if (nm.includes("tarsus")) {
          eng.tarsusIndices.push(idx);
        }
      });

      // Upgrade fly silhouette to real CT-scanned NeuroMechFly v2 Apache-2.0 STL anatomy mounted
      // on UNROTATED MuJoCo BODY poses (thorax, head, proboscis — never rotated capsule geoms!).
      if (useCtBody) {
        const curSig = sig;
        Promise.all([buildCtThoraxGroup(), buildCtHeadGroup(), buildCtProboscisGroup()]).then(
          ([ctThorax, ctHead, ctProb]) => {
            const cur = arenaRef.current;
            if (!cur || cur.geomSignature !== curSig || !ctThorax || !ctHead) return;
            for (const ph of placeholderHide) ph.visible = false;
            cur.ctMounts.thorax.add(ctThorax);
            cur.ctMounts.head.add(ctHead);
            if (ctProb) cur.ctMounts.proboscis.add(ctProb);
          },
        );
      }

      eng.geomSignature = sig;
      snapImmediate = true;
    }

    const f32 = new Float32Array(frame.bits.buffer, frame.bits.byteOffset, h.geom_count * 12);
    const mat4 = scratchMat4Ref.current;
    for (let i = 0; i < h.geom_count && i < eng.meshes.length; i++) {
      const o = i * 12;
      if (!snapImmediate) {
        eng.prevPos[i].copy(eng.meshes[i].position);
        eng.prevQuat[i].copy(eng.meshes[i].quaternion);
      }
      eng.targetPos[i].set(f32[o], f32[o + 1], f32[o + 2]);
      mat4.set(
        f32[o + 3], f32[o + 4], f32[o + 5], 0,
        f32[o + 6], f32[o + 7], f32[o + 8], 0,
        f32[o + 9], f32[o + 10], f32[o + 11], 0,
        0, 0, 0, 1,
      );
      eng.targetQuat[i].setFromRotationMatrix(mat4);
      if (snapImmediate) {
        eng.prevPos[i].copy(eng.targetPos[i]);
        eng.prevQuat[i].copy(eng.targetQuat[i]);
        eng.meshes[i].position.copy(eng.targetPos[i]);
        eng.meshes[i].quaternion.copy(eng.targetQuat[i]);
      }
    }

    // Update unrotated MuJoCo body poses for CT-scanned thorax/head/proboscis groups.
    const bp = h.body_poses;
    if (bp) {
      const snapCt = snapImmediate || !eng.ctHasPose;
      for (const k of ["thorax", "head", "proboscis"] as const) {
        const pose = bp[k];
        if (!pose) continue;
        if (!snapCt) {
          eng.ctPrevPos[k].copy(eng.ctMounts[k].position);
          eng.ctPrevQuat[k].copy(eng.ctMounts[k].quaternion);
        }
        eng.ctTargetPos[k].set(pose.pos[0], pose.pos[1], pose.pos[2]);
        const r = pose.mat;
        mat4.set(
          r[0], r[1], r[2], 0,
          r[3], r[4], r[5], 0,
          r[6], r[7], r[8], 0,
          0, 0, 0, 1,
        );
        eng.ctTargetQuat[k].setFromRotationMatrix(mat4);
        if (snapCt) {
          eng.ctPrevPos[k].copy(eng.ctTargetPos[k]);
          eng.ctPrevQuat[k].copy(eng.ctTargetQuat[k]);
          eng.ctMounts[k].position.copy(eng.ctTargetPos[k]);
          eng.ctMounts[k].quaternion.copy(eng.ctTargetQuat[k]);
        }
      }
      eng.ctHasPose = true;
    }

    // Highlight tarsus spheres green when touching floor.
    const contacts = h.sensors?.contacts ?? [];
    eng.tarsusIndices.forEach((gIdx, legIdx) => {
      const m = eng.meshes[gIdx];
      if (m instanceof THREE.Mesh && m.material instanceof THREE.MeshStandardMaterial) {
        m.material.color.copy(contacts[legIdx] ? TARSUS_DOWN : TARSUS_AIR);
      }
    });

    // Update food droplets/crystals & odor plume halos when food list changes.
    const fsig = JSON.stringify(h.foods ?? []);
    if (fsig !== eng.foodSignature) {
      while (eng.foodGroup.children.length > 0) {
        eng.foodGroup.remove(eng.foodGroup.children[0]);
      }
      for (const f of h.foods ?? []) {
        eng.foodGroup.add(makeFoodObject(f));
      }
      eng.foodSignature = fsig;
    }

    // Update footprint trail in-place using pre-allocated Float32Array (zero per-frame GC allocation).
    const trail = h.trail ?? [];
    const nPts = Math.min(trail.length, eng.trailBuf.length / 3);
    if (nPts >= 2) {
      const startIdx = trail.length - nPts;
      for (let i = 0; i < nPts; i++) {
        eng.trailBuf[i * 3] = trail[startIdx + i][0];
        eng.trailBuf[i * 3 + 1] = trail[startIdx + i][1];
        eng.trailBuf[i * 3 + 2] = 0.015;
      }
      const posAttr = eng.trailLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      posAttr.needsUpdate = true;
      eng.trailLine.geometry.setDrawRange(0, nPts);
    } else {
      eng.trailLine.geometry.setDrawRange(0, 0);
    }

    // Follow fly smoothly in rAF.
    if (h.torso) {
      const [tx, ty, tz] = h.torso;
      if (!eng.targetTorso) eng.targetTorso = new THREE.Vector3(tx, ty, tz);
      else eng.targetTorso.set(tx, ty, tz);
      if (!eng.followInitialized) {
        eng.controls.target.copy(eng.targetTorso);
        eng.camera.position.set(tx - 1.85, ty - 1.55, tz + 1.45);
        eng.followInitialized = true;
      }
    }
  }, []);

  useImperativeHandle(ref, () => ({ draw: drawFrame }), [drawFrame]);

  useEffect(() => {
    const host = arenaHostRef.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      });
    } catch {
      setWebglStatus("unavailable");
      return;
    }
    try {
      const gl = renderer.getContext();
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      const rawName = dbg
        ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) ?? "")
        : String(gl.getParameter(gl.RENDERER) ?? "");
      setGpuRenderer(rawName.replace(/^ANGLE \(([^,]+),\s*/, "").replace(/,\s*Direct3D.*$/, "").slice(0, 42));
    } catch {
      /* ignore GPU string probe failures */
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(paneWidth, paneHeight);
    renderer.setClearColor(0x0f172a, 1);
    host.innerHTML = "";
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.up.set(0, 0, 1);
    const camera = new THREE.PerspectiveCamera(44, paneWidth / paneHeight, 0.05, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(-1.35, -1.15, 1.05);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.22);

    scene.add(new THREE.AmbientLight(0xdbeafe, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 1.15);
    sun.position.set(2.5, -2.0, 4.5);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x93c5fd, 0.45);
    fill.position.set(-2.5, 2.5, 2.0);
    scene.add(fill);

    const foodGroup = new THREE.Group();
    scene.add(foodGroup);

    const ctMounts = {
      thorax: new THREE.Group(),
      head: new THREE.Group(),
      proboscis: new THREE.Group(),
    };
    scene.add(ctMounts.thorax, ctMounts.head, ctMounts.proboscis);

    const trailBuf = new Float32Array(256 * 3);
    const trailGeo = new THREE.BufferGeometry();
    const trailAttr = new THREE.BufferAttribute(trailBuf, 3);
    trailAttr.setUsage(THREE.DynamicDrawUsage);
    trailGeo.setAttribute("position", trailAttr);
    trailGeo.setDrawRange(0, 0);
    const trailLine = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.75 }),
    );
    trailLine.frustumCulled = false;
    scene.add(trailLine);

    arenaRef.current = {
      renderer,
      scene,
      camera,
      controls,
      meshes: [],
      prevPos: [],
      prevQuat: [],
      targetPos: [],
      targetQuat: [],
      ctMounts,
      ctPrevPos: { thorax: new THREE.Vector3(), head: new THREE.Vector3(), proboscis: new THREE.Vector3() },
      ctPrevQuat: { thorax: new THREE.Quaternion(), head: new THREE.Quaternion(), proboscis: new THREE.Quaternion() },
      ctTargetPos: { thorax: new THREE.Vector3(), head: new THREE.Vector3(), proboscis: new THREE.Vector3() },
      ctTargetQuat: { thorax: new THREE.Quaternion(), head: new THREE.Quaternion(), proboscis: new THREE.Quaternion() },
      ctHasPose: false,
      interpStartMs: performance.now(),
      frameIntervalMs: 18,
      lastFrameArrivalMs: 0,
      targetTorso: null,
      tarsusIndices: [],
      foodGroup,
      trailLine,
      trailBuf,
      geomSignature: "",
      foodSignature: "",
      followInitialized: false,
    };

    let raf = 0;
    let lastRafMs = performance.now();
    const loop = () => {
      raf = window.requestAnimationFrame(loop);
      const nowRaf = performance.now();
      const dtSec = Math.min(0.05, Math.max(0.001, (nowRaf - lastRafMs) * 0.001));
      lastRafMs = nowRaf;
      rafFramesRef.current += 1;
      const st = arenaRef.current;
      if (st) {
        // Constant-velocity snapshot interpolation (prevPose -> currPose over measured frame interval)
        // keeps 60-144 FPS motion completely silky at any simulation speed multiplier.
        const span = Math.max(14, Math.min(120, st.frameIntervalMs * 1.05));
        const alpha = Math.max(0, Math.min(1, (nowRaf - st.interpStartMs) / span));
        const camBlend = 1.0 - Math.exp(-8.5 * dtSec);
        for (let i = 0; i < st.meshes.length; i++) {
          st.meshes[i].position.lerpVectors(st.prevPos[i], st.targetPos[i], alpha);
          st.meshes[i].quaternion.slerpQuaternions(st.prevQuat[i], st.targetQuat[i], alpha);
        }
        if (st.ctHasPose) {
          for (const k of ["thorax", "head", "proboscis"] as const) {
            st.ctMounts[k].position.lerpVectors(st.ctPrevPos[k], st.ctTargetPos[k], alpha);
            st.ctMounts[k].quaternion.slerpQuaternions(st.ctPrevQuat[k], st.ctTargetQuat[k], alpha);
          }
        }
        if (st.targetTorso) {
          st.controls.target.lerp(st.targetTorso, camBlend);
        }
      }
      controls.update();
      renderer.render(scene, camera);
    };
    loop();

    const fpsTimer = window.setInterval(() => {
      onDisplayFps?.(framesRef.current);
      setRenderFps(rafFramesRef.current);
      framesRef.current = 0;
      rafFramesRef.current = 0;
    }, 1000);

    return () => {
      window.cancelAnimationFrame(raf);
      window.clearInterval(fpsTimer);
      controls.dispose();
      renderer.dispose();
      arenaRef.current = null;
    };
  }, [paneWidth, paneHeight, onDisplayFps]);

  const cmd = hud?.commands ?? { forward: 0, turn: 0, backward: 0, proboscis: 0, grooming: 0 };
  const tm = hud?.timing_ms;
  const rtf = tm?.realtime_factor ?? 0;
  const isJa = t("lang.switch") === "EN";

  const meters: Array<{ key: string; label: string; value: number; color: string; bipolar?: boolean }> = [
    { key: "forward", label: t("fly.cmd.forward"), value: cmd.forward, color: "#38bdf8" },
    { key: "turn", label: t("fly.cmd.turn"), value: cmd.turn, color: "#a78bfa", bipolar: true },
    { key: "backward", label: t("fly.cmd.backward"), value: cmd.backward, color: "#94a3b8" },
    { key: "proboscis", label: t("fly.cmd.proboscis"), value: cmd.proboscis, color: "#f59e0b" },
    { key: "grooming", label: t("fly.cmd.grooming"), value: cmd.grooming, color: "#34d399" },
  ];

  return (
    <div data-testid={testId} style={{ display: "flex", flexDirection: "column", gap: "10px", width: "100%" }}>
      {/* Honest Level 1 / Level 2 VNC / F12d Full-Brain Replay Banner */}
      <div
        data-testid="fly-level1-banner"
        className="notice small"
        style={{
          margin: 0,
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "8px",
          borderLeft: hud?.evolved_descendant
            ? "3px solid #d946ef"
            : hud?.control_mode && hud.control_mode !== "level1_cpg"
              ? "3px solid #10b981"
              : undefined,
        }}
      >
        <span>
          {hud?.evolved_descendant
            ? t("fly.f12d_replay_banner")
                .replace("{id}", String(hud.evolved_descendant.fly_id ?? 0))
                .replace("{gen}", String(hud.evolved_descendant.generation ?? 0))
                .replace("{dist}", hud.evolved_descendant.fly_distance.toFixed(3))
                .replace("{sugar}", hud.evolved_descendant.sugar_scale.toFixed(2))
                .replace("{bitter}", hud.evolved_descendant.bitter_scale.toFixed(2))
                .replace("{touch}", hud.evolved_descendant.touch_scale.toFixed(2))
            : hud?.vnc_profile
              ? t("fly.level2_alif_banner").replace("{profile}", hud.vnc_profile)
              : hud?.control_mode === "level2_vnc_tuned"
                ? t("fly.level2_tuned_banner")
                : hud?.control_mode === "level2_vnc"
                  ? t("fly.level2_banner")
                  : t("fly.level1_banner")}
        </span>
        {tm && (
          <span className="mono" data-testid="fly-realtime-badge">
            {t("fly.realtime").replace("{rtf}", rtf.toFixed(2))}
            {" · "}
            {t("fly.timing_short")
              .replace("{brain}", tm.brain.toFixed(2))
              .replace("{body}", tm.body.toFixed(2))
              .replace("{xfer}", tm.transfer.toFixed(2))}
          </span>
        )}
      </div>

      {/* Live Interventions Toolbar */}
      <div className="toolbar" role="group" data-testid="fly-interventions" style={{ margin: 0 }}>
        <button
          type="button"
          data-testid="btn-fly-sugar"
          disabled={!live}
          onClick={() => onIntervene("place_food", { kind: "sugar" })}
        >
          🍯 {t("fly.act.place_sugar")}
        </button>
        <button
          type="button"
          data-testid="btn-fly-bitter"
          disabled={!live}
          onClick={() => onIntervene("place_food", { kind: "bitter" })}
        >
          🧪 {t("fly.act.place_bitter")}
        </button>
        <button
          type="button"
          data-testid="btn-fly-poke"
          disabled={!live}
          onClick={() => onIntervene("poke_antenna", { duration_ms: 1200 })}
        >
          🪶 {t("fly.act.poke_antenna")}
        </button>
        <button
          type="button"
          data-testid="btn-fly-silence-mn9"
          disabled={!live}
          onClick={() =>
            onIntervene("silence", {
              port: "mn9_proboscis_extension",
              silenced: !(hud?.mn9_silenced ?? false),
            })
          }
        >
          {hud?.mn9_silenced ? `🔊 ${t("fly.act.unsilence_mn9")}` : `🔇 ${t("fly.act.silence_mn9")}`}
        </button>
        <button
          type="button"
          data-testid="btn-fly-clear-food"
          disabled={!live}
          onClick={() => onIntervene("clear_food", {})}
        >
          🧹 {t("fly.act.clear_food")}
        </button>
      </div>

      {/* Side-by-Side 2-Pane View: Left 3D Arena | Right 3D Glowing Brain */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "10px",
          alignItems: "start",
        }}
      >
        <div
          data-testid="fly-arena-pane"
          style={{
            border: "1px solid var(--border, #334155)",
            borderRadius: "8px",
            overflow: "hidden",
            background: "#0f172a",
          }}
        >
          <div
            style={{
              padding: "6px 10px",
              fontSize: "0.82rem",
              display: "flex",
              justifyContent: "space-between",
              background: "rgba(15,23,42,0.92)",
              color: "#e2e8f0",
              borderBottom: "1px solid #1e293b",
            }}
          >
            <strong>{t("fly.pane.arena")}</strong>
            <span className="mono" style={{ opacity: 0.85 }}>
              {renderFps > 0 ? `${renderFps} FPS` : ""}
              {gpuRenderer ? ` · ${gpuRenderer}` : ""}
              {" · "}
              {hud?.wiring === "shuffled_control" ? t("form.connectome.wiring.shuffled_control") : t("form.connectome.wiring.reference")}
            </span>
          </div>
          {webglStatus === "unavailable" ? (
            <div style={{ padding: "24px", color: "#94a3b8", textAlign: "center" }}>
              WebGL unavailable ({hud ? `t=${hud.tick}` : "waiting"})
            </div>
          ) : (
            <div ref={arenaHostRef} style={{ width: "100%", height: `${paneHeight}px`, overflow: "hidden" }} />
          )}
          {/* Body-part circuit explorer buttons */}
          <div
            style={{
              padding: "6px 10px",
              display: "flex",
              gap: "6px",
              flexWrap: "wrap",
              alignItems: "center",
              background: "#111827",
              borderTop: "1px solid #1e293b",
              fontSize: "0.8rem",
            }}
          >
            <span style={{ color: "#94a3b8" }}>{t("fly.part.prompt")}:</span>
            {(["proboscis", "antennae", "legs"] as const).map((k) => (
              <button
                key={k}
                type="button"
                className={selectedPart === k ? "primary" : ""}
                style={{ padding: "2px 8px", fontSize: "0.78rem" }}
                onClick={() => setSelectedPart(selectedPart === k ? null : k)}
              >
                {t(`fly.part.${k}`)}
              </button>
            ))}
          </div>
          {selectedPart && (
            <div
              data-testid="fly-part-link"
              style={{
                padding: "8px 10px",
                background: "#1e293b",
                color: "#f1f5f9",
                fontSize: "0.8rem",
                borderTop: "1px solid #334155",
              }}
            >
              <strong>{isJa ? BODY_LINKS[selectedPart].labelJa : BODY_LINKS[selectedPart].labelEn}</strong>
              <p style={{ margin: "4px 0 0", opacity: 0.9 }}>
                {isJa ? BODY_LINKS[selectedPart].detailJa : BODY_LINKS[selectedPart].detailEn}
              </p>
            </div>
          )}
        </div>

        <div
          data-testid="fly-brain-pane"
          style={{
            border: "1px solid var(--border, #334155)",
            borderRadius: "8px",
            overflow: "hidden",
            background: "#0b1120",
          }}
        >
          <div
            style={{
              padding: "6px 10px",
              fontSize: "0.82rem",
              display: "flex",
              justifyContent: "space-between",
              background: "rgba(15,23,42,0.92)",
              color: "#e2e8f0",
              borderBottom: "1px solid #1e293b",
            }}
          >
            <strong>{hud?.has_vnc_ports ? t("fly.pane.cns") : t("fly.pane.brain")}</strong>
            <span className="mono" style={{ opacity: 0.85 }}>
              {hud ? `${hud.num_cells.toLocaleString()} cells · ${hud.fired_count} spikes` : datasetId}
            </span>
          </div>
          <Suspense fallback={<div style={{ padding: "24px", color: "#94a3b8" }}>3D Brain…</div>}>
            <Brain3D
              ref={attachBrain}
              datasetId={hud?.dataset_id ?? datasetId}
              source={frameSource}
              width={paneWidth}
              testId="fly-brain-3d"
            />
          </Suspense>
        </div>
      </div>

      {/* Live Descending / Motor Command Meters */}
      <div
        data-testid="fly-command-meters"
        className="panel"
        style={{ margin: 0, padding: "10px 12px" }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", fontSize: "0.85rem" }}>
          <strong>{t("fly.commands.title")}</strong>
          <span className="small muted mono">
            {hud
              ? `sugar=${hud.sensors.sugar.toFixed(2)} · bitter=${hud.sensors.bitter.toFixed(2)} · touch=${hud.sensors.antennal_touch.toFixed(1)} · odor L/R=${hud.sensors.odor_left.toFixed(2)}/${hud.sensors.odor_right.toFixed(2)}`
              : ""}
          </span>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
            gap: "8px 14px",
          }}
        >
          {meters.map((m) => {
            const pct = m.bipolar ? Math.min(100, Math.abs(m.value) * 100) : Math.min(100, Math.max(0, m.value) * 100);
            return (
              <div key={m.key} data-testid={`fly-meter-${m.key}`}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: "2px" }}>
                  <span>{m.label}</span>
                  <span className="mono">{m.bipolar ? `${m.value >= 0 ? "+" : ""}${m.value.toFixed(2)}` : `${Math.round(pct)}%`}</span>
                </div>
                <div style={{ height: "8px", borderRadius: "4px", background: "#1e293b", overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${pct}%`,
                      height: "100%",
                      background: m.color,
                      transition: "width 80ms linear",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Phase F8 Level 2: Live 6-Leg x 6-Muscle VNC Motor Neuron & Proprioception Matrix */}
      {hud?.has_vnc_ports && hud.leg_mn_hz && (
        <div data-testid="fly-vnc-matrix" className="panel" style={{ margin: 0, padding: "10px 12px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px", marginBottom: "8px", fontSize: "0.85rem" }}>
            <strong>{t("fly.vnc.title")}</strong>
            <span className="small muted mono">
              MN avg={(hud.vnc_mn_mean_hz ?? 0).toFixed(1)} Hz · Tripod alt={(hud.vnc_tripod_alternation ?? 0).toFixed(2)}
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "8px" }}>
            {(["LF", "LM", "LH", "RF", "RM", "RH"] as const).map((leg, li) => {
              const row = hud.leg_mn_hz?.[leg] ?? {};
              const prop = hud.proprio?.[li] ?? 0;
              const channels: Array<[string, string, string]> = [
                ["coxa_pro", "Cx+", "#38bdf8"],
                ["coxa_ret", "Cx-", "#0284c7"],
                ["fem_flex", "Fe+", "#34d399"],
                ["fem_ext", "Fe-", "#059669"],
                ["tib_ext", "Ti+", "#f59e0b"],
                ["tib_flex", "Ti-", "#d97706"],
              ];
              return (
                <div
                  key={leg}
                  style={{
                    background: "rgba(15,23,42,0.55)",
                    border: "1px solid #1e293b",
                    borderRadius: "6px",
                    padding: "6px 8px",
                    fontSize: "0.75rem",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                    <strong className="mono">{leg}</strong>
                    <span className="mono muted">prop {(prop * 100).toFixed(0)}%</span>
                  </div>
                  {channels.map(([ck, lbl, col]) => {
                    const hz = row[ck] ?? 0;
                    const pct = Math.min(100, (hz / 40) * 100);
                    return (
                      <div key={ck} style={{ display: "grid", gridTemplateColumns: "28px 1fr 38px", alignItems: "center", gap: "4px", marginBottom: "2px" }}>
                        <span className="mono muted">{lbl}</span>
                        <div style={{ height: "5px", borderRadius: "3px", background: "#1e293b", overflow: "hidden" }}>
                          <div style={{ width: `${pct}%`, height: "100%", background: col }} />
                        </div>
                        <span className="mono" style={{ textAlign: "right" }}>{hz.toFixed(0)}Hz</span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
});
