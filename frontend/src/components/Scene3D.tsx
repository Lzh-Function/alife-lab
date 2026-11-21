// 3D view of embodied creatures (codec scene3d_f32): MuJoCo geoms as three.js meshes, z up.
// The header lists the geoms (type, size, rgba); the payload holds per geom position (3) and
// rotation matrix (9) as float32. The camera orbits (mouse) and follows the torso.
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Frame } from "../lib/api";

export interface Scene3DHandle {
  draw: (frame: Frame) => void;
}

interface GeomInfo {
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
}

function meshFor(g: GeomInfo): THREE.Object3D {
  const colour = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
  const mat = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.6, metalness: 0.05 });
  let geo: THREE.BufferGeometry;
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

export const Scene3D = forwardRef<Scene3DHandle, Props>(function Scene3D({ onDisplayFps, onClick, testId = "world-canvas", width = 720, height = 520 }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const three = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    meshes: THREE.Object3D[];
    signature: string;
    lastTorso: THREE.Vector3 | null;
  } | null>(null);
  const frames = useRef(0);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
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
    controls.target.set(0, 0, 0.3);
    controls.update();
    three.current = { renderer, scene, camera, controls, meshes: [], signature: "", lastTorso: null };
    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
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
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      three.current = null;
    };
  }, [width, height, onDisplayFps, testId]);

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => {
      const t = three.current;
      if (!t || frame.header.codec !== "scene3d_f32") return;
      const header = frame.header as unknown as { geoms: GeomInfo[]; count: number; torso: number[] };
      const signature = JSON.stringify(header.geoms);
      if (signature !== t.signature) {
        for (const m of t.meshes) t.scene.remove(m);
        t.meshes = header.geoms.map((g) => {
          const m = meshFor(g);
          m.matrixAutoUpdate = false;
          t.scene.add(m);
          return m;
        });
        t.signature = signature;
      }
      // The payload may start at any byte offset inside the message; copy to an aligned buffer.
      const f = new Float32Array(frame.bits.slice().buffer);
      header.geoms.forEach((g, i) => {
        const o = i * 12;
        const m = t.meshes[i];
        if (g.type === "plane") {
          m.matrix.identity();
        } else {
          // MuJoCo xmat is row-major; Matrix4.set takes row-major arguments.
          m.matrix.set(f[o + 3], f[o + 4], f[o + 5], f[o], f[o + 6], f[o + 7], f[o + 8], f[o + 1], f[o + 9], f[o + 10], f[o + 11], f[o + 2], 0, 0, 0, 1);
        }
        m.matrixWorldNeedsUpdate = true;
      });
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
