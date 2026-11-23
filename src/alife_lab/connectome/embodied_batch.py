"""Phase F9: Parallel Multi-Fly Simulation (MuJoCo Warp GPU + Batched GPU LIF).

Couples ``CupyBatchedLIFRunner`` (B independent whole-brain spiking connectomes sharing one CSR
graph on GPU) with ``mujoco_warp`` (B parallel 6-legged fly bodies stepped simultaneously on GPU
via CUDA Graph capture), with an exact CPU float64 MuJoCo + NumPy fallback for offline CPU CI
and deterministic verification of GPU-discovered candidates.

Honest scientific note (connectome_plan.md §4 F9):
- GPU MuJoCo Warp uses float32 parallel constraint solving and is NOT strictly bitwise-identical
  across GPU runs (~1.5e-8 float32 ULP) or vs CPU float64 MuJoCo (~2e-7 qpos over 50 ms).
- Every screening or search function therefore supports automatic CPU float64 re-verification of
  top findings and records the exact ``gpu_vs_cpu`` behavioral difference.
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.analysis.stats import compare, describe, holm
from alife_lab.connectome.embodied import (
    EmbodiedPorts,
    make_embodied_world,
    resolve_embodied_ports,
    step_embodied_tick,
)
from alife_lab.connectome.fly_body import (
    FoodSpot,
    build_fly_mjcf,
)
from alife_lab.connectome.lif import (
    LIFParams,
    LIFState,
    make_initial_state,
    make_shuffled_csr,
    step_tick_cpu,
)
from alife_lab.connectome.schema import ConnectomePack, CSRGraph

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
U8 = npt.NDArray[np.uint8]

_WARP_INIT_DONE = False


def warp_available() -> bool:
    """True if ``mujoco_warp`` and ``warp`` are installed and a CUDA GPU is usable."""
    if importlib.util.find_spec("mujoco_warp") is None or importlib.util.find_spec("warp") is None:
        return False
    try:
        _ensure_warp_init()
        import warp as wp

        return bool(wp.is_cuda_available())
    except Exception:
        return False


def _ensure_warp_init() -> None:
    global _WARP_INIT_DONE
    if _WARP_INIT_DONE:
        return
    if "WARP_CACHE_DIR" not in os.environ:
        default = os.path.expanduser("~/.cache/warp")
        parent = os.path.dirname(default) or "/"
        if not os.access(default if os.path.exists(default) else parent, os.W_OK):
            os.environ["WARP_CACHE_DIR"] = os.path.join(
                tempfile.gettempdir(), f"alife_warp_cache_{os.getuid()}"
            )
    import warp

    wp: Any = warp
    wp.config.quiet = True
    wp.init()
    _WARP_INIT_DONE = True


def _resolve_backends(num_cells: int, physics_backend: str, brain_backend: str) -> tuple[str, str]:
    p_back = physics_backend
    if p_back == "auto":
        p_back = "warp" if warp_available() else "cpu"
    b_back = brain_backend
    if b_back == "auto":
        from alife_lab.connectome.lif_cupy import available as cupy_ok

        b_back = "cupy" if (num_cells > 500 and cupy_ok()) else "numpy"
    if p_back not in ("warp", "cpu"):
        raise ValueError(f"unsupported physics_backend {p_back!r}; choose 'auto', 'warp', or 'cpu'")
    if b_back not in ("cupy", "numpy"):
        raise ValueError(f"unsupported brain_backend {b_back!r}; choose 'auto', 'cupy', or 'numpy'")
    return p_back, b_back


@dataclass
class BatchedFlyOutcome:
    """Per-fly behavioral summary returned by a batched embodied rollout."""

    index: int
    seed: int
    label: str
    distance_cm: float
    net_x_cm: float
    net_y_cm: float
    min_food_dist_cm: float
    feeding_ms: int
    grooming_ms: int
    max_proboscis_cmd: float
    max_grooming_cmd: float
    mn9_mean_hz: float
    dng84_mean_hz: float


class BatchedEmbodiedRunner:
    """Step ``B`` independent embodied flies in lockstep (GPU MuJoCo Warp + GPU Batched LIF).

    Supports both ``physics_backend="warp"`` (CUDA Graph captured 10-substep 1 ms ticks across
    ``nworld = B``) and ``physics_backend="cpu"`` (exact float64 MuJoCo array of ``B`` worlds),
    and both ``brain_backend="cupy"`` (``CupyBatchedLIFRunner``) and ``brain_backend="numpy"``.
    """

    def __init__(
        self,
        pack: ConnectomePack,
        csr: CSRGraph,
        *,
        seeds: list[int],
        foods: list[FoodSpot] | None = None,
        poke_ms: int = 0,
        explore_drive_hz: F32 | float = 110.0,
        turn_gain: F32 | float = 1.0,
        silenced_mask: U8 | None = None,
        physics_backend: str = "auto",
        brain_backend: str = "auto",
        arena_half_size: float = 6.0,
    ) -> None:
        import mujoco

        self.pack = pack
        self.csr = csr
        self.ports: EmbodiedPorts = resolve_embodied_ports(pack)
        self.params = LIFParams()
        self.seeds = [int(s) for s in seeds]
        self.batch_size = len(self.seeds)
        if self.batch_size < 1:
            raise ValueError("seeds must contain at least one seed")
        self.foods = list(foods) if foods is not None else []
        self.arena_half_size = float(arena_half_size)
        self.poke_remaining = np.full(self.batch_size, int(poke_ms), dtype=np.int32)

        self.explore_drive_hz = (
            np.full(self.batch_size, float(explore_drive_hz), dtype=np.float32)
            if isinstance(explore_drive_hz, (int, float))
            else np.asarray(explore_drive_hz, dtype=np.float32).reshape(self.batch_size)
        )
        self.turn_gain = (
            np.full(self.batch_size, float(turn_gain), dtype=np.float32)
            if isinstance(turn_gain, (int, float))
            else np.asarray(turn_gain, dtype=np.float32).reshape(self.batch_size)
        )

        self.physics_backend, self.brain_backend = _resolve_backends(
            pack.cells.num_cells, physics_backend, brain_backend
        )

        # Build shared MuJoCo fly model
        self.mj_model = mujoco.MjModel.from_xml_string(build_fly_mjcf())
        base_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, base_data)
        self.base_qpos = base_data.qpos.copy()
        self.base_qvel = base_data.qvel.copy()

        self.tarsus_geom_ids = np.array(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"g_{leg}_tarsus")
                for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
            ],
            dtype=np.int32,
        )
        self.head_body_id = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "head"))
        self.leg_sides = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float64)
        self.tripod_offsets = np.array([0.0, math.pi, 0.0, math.pi, 0.0, math.pi], dtype=np.float64)

        # Physics runtime (GPU MuJoCo Warp OR CPU float64 MuJoCo list)
        self._warp_model: Any = None
        self._warp_data: Any = None
        self._warp_graph: Any = None
        self._cpu_datas: list[Any] = []
        if self.physics_backend == "warp":
            _ensure_warp_init()
            import mujoco_warp as mjw
            import warp

            wp: Any = warp
            self._warp_model = mjw.put_model(self.mj_model)
            self._warp_data = mjw.put_data(self.mj_model, base_data, nworld=self.batch_size)
            # Warmup + capture 10 substeps (1.0 ms) as a single CUDA Graph!
            mjw.step(self._warp_model, self._warp_data)
            wp.synchronize()
            # Reset back to exact initial state after warmup
            self._warp_data = mjw.put_data(self.mj_model, base_data, nworld=self.batch_size)
            with wp.ScopedCapture() as cap:
                for _ in range(10):
                    mjw.step(self._warp_model, self._warp_data)
            self._warp_graph = cap.graph
            wp.synchronize()
        else:
            for _ in range(self.batch_size):
                d = mujoco.MjData(self.mj_model)
                mujoco.mj_forward(self.mj_model, d)
                self._cpu_datas.append(d)

        # Brain runtime (GPU CupyBatchedLIFRunner OR CPU LIFState list)
        n = pack.cells.num_cells
        init_stim = np.zeros((self.batch_size, n), dtype=np.float32)
        sil = (
            np.zeros((self.batch_size, n), dtype=np.uint8)
            if silenced_mask is None
            else np.ascontiguousarray(silenced_mask, dtype=np.uint8)
        )
        self._gpu_brain: Any = None
        self._cpu_brains: list[LIFState] = []
        if self.brain_backend == "cupy":
            from alife_lab.connectome.lif_cupy import CupyBatchedLIFRunner

            self._gpu_brain = CupyBatchedLIFRunner(csr, self.params, self.seeds, init_stim)
            self._gpu_brain.set_silenced(sil)
        else:
            for i in range(self.batch_size):
                st = make_initial_state(n, self.params)
                st.silenced[...] = sil[i]
                self._cpu_brains.append(st)

        # Precompute index map inside all_sensory_union so per-tick uploads are tiny (B, K_sens).
        self.sens_union = self.ports.all_sensory_union
        sens_pos = {int(c): k for k, c in enumerate(self.sens_union.tolist())}

        def _loc(arr: I32) -> I32:
            return np.asarray(
                [sens_pos[int(c)] for c in arr.tolist() if int(c) in sens_pos], dtype=np.int32
            )

        self._s_sugar = _loc(self.ports.sugar_grn)
        self._s_bitter = _loc(self.ports.bitter_grn)
        self._s_groom = _loc(self.ports.groom_mech)
        self._s_olf_l = _loc(self.ports.odor_left)
        self._s_olf_r = _loc(self.ports.odor_right)
        self._s_fwd = _loc(self.ports.forward_dn)
        self._s_tl = _loc(self.ports.turn_left_dn)
        self._s_tr = _loc(self.ports.turn_right_dn)
        self._stim_buf = np.zeros((self.batch_size, self.sens_union.size), dtype=np.float32)

        # Command spike tracking (delta on cumulative total_spikes, masking silenced output MNs!)
        self.cmd_union = self.ports.all_command_union
        self._cmd_active_mask = (sil[:, self.cmd_union] == 0).astype(np.int32)
        self._prev_cmd_spikes = np.zeros((self.batch_size, self.cmd_union.size), dtype=np.int32)
        self.rates = np.zeros(
            (self.batch_size, 6), dtype=np.float64
        )  # fwd, tl, tr, bwd, prob, groom

        # Batched CPG phases & metrics
        self.cpg_phase = np.zeros(self.batch_size, dtype=np.float64)
        self.groom_phase = np.zeros(self.batch_size, dtype=np.float64)
        self.ctrl_buf = np.zeros((self.batch_size, self.mj_model.nu), dtype=np.float32)
        self.prev_xy = np.tile(self.base_qpos[:2], (self.batch_size, 1))
        self.init_xy = self.prev_xy.copy()
        self.distance_cm = np.zeros(self.batch_size, dtype=np.float64)
        self.min_food_dist = np.full(self.batch_size, 999.0, dtype=np.float64)
        self.feeding_ms = np.zeros(self.batch_size, dtype=np.int32)
        self.grooming_ms = np.zeros(self.batch_size, dtype=np.int32)
        self.max_proboscis = np.zeros(self.batch_size, dtype=np.float64)
        self.max_grooming = np.zeros(self.batch_size, dtype=np.float64)
        self.mn9_spike_sum = np.zeros(self.batch_size, dtype=np.int64)
        self.dng84_spike_sum = np.zeros(self.batch_size, dtype=np.int64)

        self.tick = 0
        self.brain_ms_total = 0.0
        self.body_ms_total = 0.0
        self.xfer_ms_total = 0.0

    def _read_poses(self) -> tuple[F64, F64, F64]:
        """Return (x[B], y[B], yaw[B]) across all B flies."""
        if self.physics_backend == "warp":
            q = self._warp_data.qpos.numpy()  # (B, nq)
            x = q[:, 0].astype(np.float64)
            y = q[:, 1].astype(np.float64)
            qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)).astype(
                np.float64
            )
            return x, y, yaw
        xs = np.empty(self.batch_size, dtype=np.float64)
        ys = np.empty(self.batch_size, dtype=np.float64)
        yaws = np.empty(self.batch_size, dtype=np.float64)
        for i, d in enumerate(self._cpu_datas):
            q = d.qpos
            xs[i], ys[i] = float(q[0]), float(q[1])
            qw, qx, qy, qz = float(q[3]), float(q[4]), float(q[5]), float(q[6])
            yaws[i] = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return xs, ys, yaws

    def step_1ms(self) -> None:
        """Advance all B flies by one 1.0 ms closed-loop tick (10 physics + 10 LIF substeps)."""
        t0 = time.perf_counter()
        x, y, yaw = self._read_poses()
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Approximate head & bilateral antennae positions from thorax pose (exact geometry match)
        hx = x + 0.28 * cy
        hy = y + 0.28 * sy
        ant_lx = hx + 0.08 * cy - 0.14 * sy
        ant_ly = hy + 0.08 * sy + 0.14 * cy
        ant_rx = hx + 0.08 * cy + 0.14 * sy
        ant_ry = hy + 0.08 * sy - 0.14 * cy

        sugar = np.zeros(self.batch_size, dtype=np.float64)
        bitter = np.zeros(self.batch_size, dtype=np.float64)
        odor_l = np.zeros(self.batch_size, dtype=np.float64)
        odor_r = np.zeros(self.batch_size, dtype=np.float64)
        for f in self.foods:
            d_head = np.hypot(hx - f.x, hy - f.y)
            d_body = np.hypot(x - f.x, y - f.y)
            if f.kind == "sugar":
                self.min_food_dist = np.minimum(self.min_food_dist, d_body)
                sig = np.where(
                    d_head <= f.radius, 1.0, np.where(d_body <= f.radius + 0.25, 0.85, 0.0)
                )
                sugar = np.maximum(sugar, sig)
                dl = np.hypot(ant_lx - f.x, ant_ly - f.y)
                dr = np.hypot(ant_rx - f.x, ant_ry - f.y)
                odor_l = np.maximum(odor_l, np.exp(-0.75 * dl * dl))
                odor_r = np.maximum(odor_r, np.exp(-0.75 * dr * dr))
            else:
                sig = np.where(
                    d_head <= f.radius, 1.0, np.where(d_body <= f.radius + 0.25, 0.85, 0.0)
                )
                bitter = np.maximum(bitter, sig)

        touch = (self.poke_remaining > 0).astype(np.float64)
        self.poke_remaining = np.maximum(0, self.poke_remaining - 1)
        # Wall touch triggers antennal mechanosensation too
        lim = self.arena_half_size - 0.45
        wall_hit = (np.abs(hx) >= lim) | (np.abs(hy) >= lim)
        touch = np.maximum(touch, wall_hit.astype(np.float64) * 0.8)

        # Build sensory upload buffer (B, K_sens)
        self._stim_buf.fill(np.float32(0.0))
        if self._s_sugar.size > 0:
            self._stim_buf[:, self._s_sugar] = (180.0 * sugar)[:, None].astype(np.float32)
        if self._s_bitter.size > 0:
            self._stim_buf[:, self._s_bitter] = (200.0 * bitter)[:, None].astype(np.float32)
        if self._s_groom.size > 0:
            self._stim_buf[:, self._s_groom] = (180.0 * touch)[:, None].astype(np.float32)

        is_synthetic = self.pack.dataset_id.startswith("synthetic_")
        if is_synthetic:
            if self._s_olf_l.size > 0:
                self._stim_buf[:, self._s_olf_l] = (140.0 * odor_l)[:, None].astype(np.float32)
            if self._s_olf_r.size > 0:
                self._stim_buf[:, self._s_olf_r] = (140.0 * odor_r)[:, None].astype(np.float32)
        else:
            odor_diff = (odor_l - odor_r) * self.turn_gain
            if self._s_tl.size > 0:
                self._stim_buf[:, self._s_tl] = np.clip(450.0 * odor_diff, 0.0, 200.0)[
                    :, None
                ].astype(np.float32)
            if self._s_tr.size > 0:
                self._stim_buf[:, self._s_tr] = np.clip(-450.0 * odor_diff, 0.0, 200.0)[
                    :, None
                ].astype(np.float32)

        walk_arousal = np.where((sugar < 0.35) & (touch < 0.3), self.explore_drive_hz, 0.0)
        if self._s_fwd.size > 0:
            self._stim_buf[:, self._s_fwd] = walk_arousal[:, None].astype(np.float32)

        t_xfer0 = time.perf_counter()
        if self.brain_backend == "cupy":
            self._gpu_brain.upload_stim_subset_batch(self.sens_union, self._stim_buf)
        else:
            for i, st in enumerate(self._cpu_brains):
                st.stim_rate_hz[self.sens_union] = self._stim_buf[i]

        t_brain0 = time.perf_counter()
        if self.brain_backend == "cupy":
            self._gpu_brain.step_substeps(self.params.substeps_per_tick, use_graph=True)
            cur_totals = self._gpu_brain.read_total_spikes_subset(self.cmd_union)
        else:
            cur_totals = np.empty((self.batch_size, self.cmd_union.size), dtype=np.int32)
            for i, st in enumerate(self._cpu_brains):
                step_tick_cpu(st, self.csr, self.params, self.seeds[i])
                cur_totals[i] = st.total_spikes[self.cmd_union]
        t_brain1 = time.perf_counter()

        # Convert per-tick command spike deltas -> exponential low-pass rates (tau = 20 ms)
        delta_spk = (cur_totals - self._prev_cmd_spikes) * self._cmd_active_mask
        self._prev_cmd_spikes = cur_totals
        alpha = 1.0 - math.exp(-1.0 / 20.0)
        for ci, cname in enumerate(
            ("forward", "turn_left", "turn_right", "backward", "proboscis", "grooming")
        ):
            lo, hi = self.ports.cmd_slices[cname]
            if hi > lo:
                inst_hz = delta_spk[:, lo:hi].mean(axis=1) * 1000.0
                self.rates[:, ci] = (1.0 - alpha) * self.rates[:, ci] + alpha * inst_hz
                if cname == "proboscis":
                    self.mn9_spike_sum += delta_spk[:, lo:hi].sum(axis=1)
                elif cname == "grooming":
                    self.dng84_spike_sum += delta_spk[:, lo:hi].sum(axis=1)

        fwd = np.clip(self.rates[:, 0] / 40.0, 0.0, 1.0)
        turn = np.clip((self.rates[:, 1] - self.rates[:, 2]) / 35.0, -1.0, 1.0)
        bwd = np.clip(self.rates[:, 3] / 40.0, 0.0, 1.0)
        prob = np.clip(self.rates[:, 4] / 25.0, 0.0, 1.0)
        groom = np.clip(self.rates[:, 5] / 25.0, 0.0, 1.0)

        # Behavior arbitration & batched Level 1 CPG actuator computation
        t_body0 = time.perf_counter()
        is_groom = groom >= 0.30
        is_feed = (prob >= 0.30) & (~is_groom)
        walk_mag = np.where(
            is_groom | is_feed, 0.0, np.maximum(fwd, np.maximum(bwd, np.abs(turn) * 0.6))
        )
        fwd_net = np.where(is_groom | is_feed, 0.0, fwd - bwd)

        self.cpg_phase = (self.cpg_phase + 2.0 * math.pi * (4.0 + 7.0 * walk_mag) * 1e-3) % (
            2.0 * math.pi
        )
        self.groom_phase = np.where(
            is_groom,
            (self.groom_phase + 2.0 * math.pi * 6.5 * 1e-3) % (2.0 * math.pi),
            self.groom_phase,
        )

        self.ctrl_buf.fill(np.float32(0.0))
        for li in range(6):
            s = self.leg_sides[li]
            phi = self.cpg_phase + self.tripod_offsets[li]
            swing = np.maximum(0.0, np.sin(phi))
            side_turn = 1.0 - 0.55 * s * turn
            yaw_amp = 0.52 * walk_mag * np.clip(side_turn, 0.15, 1.35)
            lift = 0.45 * swing * walk_mag
            b = li * 3
            cx = s * yaw_amp * np.cos(phi) * np.where(fwd_net < -0.1, -1.0, 1.0)
            fm = 0.16 - lift
            tb = -0.10 + 0.35 * swing * walk_mag
            if li in (0, 3):  # front legs sweep antennae during grooming
                gp = self.groom_phase + (0.0 if li == 0 else math.pi * 0.65)
                cx = np.where(is_groom, -s * (0.48 + 0.18 * np.sin(gp)), cx)
                fm = np.where(is_groom, -0.35 - 0.12 * np.cos(gp), fm)
                tb = np.where(is_groom, 0.55 + 0.25 * np.sin(gp), tb)
            self.ctrl_buf[:, b + 0] = cx.astype(np.float32)
            self.ctrl_buf[:, b + 1] = fm.astype(np.float32)
            self.ctrl_buf[:, b + 2] = tb.astype(np.float32)

        self.ctrl_buf[:, 18] = np.where(is_groom, 0.0, 0.95 * prob).astype(np.float32)
        self.ctrl_buf[:, 19] = np.where(is_groom, 0.30 * groom, 0.14 * prob).astype(np.float32)

        # Step physics 10 substeps (1.0 ms)
        if self.physics_backend == "warp":
            import warp

            wp: Any = warp
            self._warp_data.ctrl.assign(self.ctrl_buf)
            wp.capture_launch(self._warp_graph)
            wp.synchronize()
        else:
            import mujoco

            for i, d in enumerate(self._cpu_datas):
                d.ctrl[:] = self.ctrl_buf[i]
                for _ in range(10):
                    mujoco.mj_step(self.mj_model, d)
        t_end = time.perf_counter()

        # Update behavioral tallies
        x2, y2, _ = self._read_poses()
        self.distance_cm += np.hypot(x2 - self.prev_xy[:, 0], y2 - self.prev_xy[:, 1])
        self.prev_xy[:, 0] = x2
        self.prev_xy[:, 1] = y2
        self.feeding_ms += is_feed.astype(np.int32)
        self.grooming_ms += is_groom.astype(np.int32)
        self.max_proboscis = np.maximum(self.max_proboscis, prob)
        self.max_grooming = np.maximum(self.max_grooming, groom)

        self.tick += 1
        self.xfer_ms_total += (t_brain0 - t_xfer0) * 1e3
        self.brain_ms_total += (t_brain1 - t_brain0) * 1e3
        self.body_ms_total += ((t_end - t_body0) + (t_xfer0 - t0)) * 1e3

    def run(self, ticks: int, labels: list[str] | None = None) -> list[BatchedFlyOutcome]:
        for _ in range(ticks):
            self.step_1ms()
        x_end, y_end, _ = self._read_poses()
        n_mn9 = max(1, int(self.ports.proboscis_mn.size))
        n_dng = max(1, int(self.ports.grooming_dn.size))
        dur_s = max(1e-3, self.tick * 1e-3)
        out: list[BatchedFlyOutcome] = []
        for i in range(self.batch_size):
            out.append(
                BatchedFlyOutcome(
                    index=i,
                    seed=self.seeds[i],
                    label=labels[i] if labels else f"fly_{i}",
                    distance_cm=round(float(self.distance_cm[i]), 4),
                    net_x_cm=round(float(x_end[i] - self.init_xy[i, 0]), 4),
                    net_y_cm=round(float(y_end[i] - self.init_xy[i, 1]), 4),
                    min_food_dist_cm=(
                        round(float(self.min_food_dist[i]), 4)
                        if self.min_food_dist[i] < 900.0
                        else -1.0
                    ),
                    feeding_ms=int(self.feeding_ms[i]),
                    grooming_ms=int(self.grooming_ms[i]),
                    max_proboscis_cmd=round(float(self.max_proboscis[i]), 4),
                    max_grooming_cmd=round(float(self.max_grooming[i]), 4),
                    mn9_mean_hz=round(float(self.mn9_spike_sum[i]) / (n_mn9 * dur_s), 2),
                    dng84_mean_hz=round(float(self.dng84_spike_sum[i]) / (n_dng * dur_s), 2),
                )
            )
        return out

    def timing_summary(self) -> dict[str, Any]:
        tot_ms = max(1e-3, self.brain_ms_total + self.body_ms_total + self.xfer_ms_total)
        sim_fly_s = self.batch_size * self.tick * 1e-3
        return {
            "batch_size": self.batch_size,
            "ticks": self.tick,
            "physics_backend": self.physics_backend,
            "brain_backend": self.brain_backend,
            "wall_seconds": round(tot_ms * 1e-3, 3),
            "fly_seconds_per_wall_second": round(sim_fly_s / (tot_ms * 1e-3), 2),
            "ms_per_batched_tick": round(tot_ms / max(1, self.tick), 3),
            "brain_ms_per_tick": round(self.brain_ms_total / max(1, self.tick), 3),
            "body_ms_per_tick": round(self.body_ms_total / max(1, self.tick), 3),
            "xfer_ms_per_tick": round(self.xfer_ms_total / max(1, self.tick), 3),
            "brain_gpu_mib": (
                round(self._gpu_brain.device_bytes() / (1024 * 1024), 2)
                if self._gpu_brain is not None
                else 0.0
            ),
        }


def benchmark_batched_embodied(
    pack: ConnectomePack,
    *,
    batch_sizes: tuple[int, ...] = (1, 8, 16, 32),
    ticks: int = 250,
    seed: int = 1,
    physics_backend: str = "auto",
    brain_backend: str = "auto",
) -> dict[str, Any]:
    """Measure coupled multi-fly throughput + exact CPU float64 vs GPU Warp behavioral agreement."""
    csr = pack.edges.to_csr(pack.cells.num_cells)
    p_back, b_back = _resolve_backends(pack.cells.num_cells, physics_backend, brain_backend)
    foods = [FoodSpot(x=0.18, y=0.0, radius=0.45, kind="sugar")]

    rows: list[dict[str, Any]] = []
    for b in batch_sizes:
        r = BatchedEmbodiedRunner(
            pack,
            csr,
            seeds=[seed + i for i in range(b)],
            foods=foods,
            physics_backend=p_back,
            brain_backend=b_back,
        )
        outs = r.run(ticks)
        tm = r.timing_summary()
        tm["mean_feeding_ms"] = round(float(np.mean([o.feeding_ms for o in outs])), 1)
        tm["mean_distance_cm"] = round(float(np.mean([o.distance_cm for o in outs])), 4)
        rows.append(tm)

    # Compare GPU Warp (nworld=1) against exact CPU float64 MuJoCo on identical seed & brain!
    cpu_ref = BatchedEmbodiedRunner(
        pack, csr, seeds=[seed], foods=foods, physics_backend="cpu", brain_backend=b_back
    )
    cpu_out = cpu_ref.run(ticks)[0]
    gpu_out: BatchedFlyOutcome | None = None
    if p_back == "warp":
        gpu_ref = BatchedEmbodiedRunner(
            pack, csr, seeds=[seed], foods=foods, physics_backend="warp", brain_backend=b_back
        )
        gpu_out = gpu_ref.run(ticks)[0]

    return {
        "dataset_id": pack.dataset_id,
        "num_cells": pack.cells.num_cells,
        "physics_backend": p_back,
        "brain_backend": b_back,
        "ticks": ticks,
        "scaling": rows,
        "cpu_vs_warp_check": {
            "cpu_feeding_ms": cpu_out.feeding_ms,
            "cpu_mn9_hz": cpu_out.mn9_mean_hz,
            "cpu_distance_cm": cpu_out.distance_cm,
            "warp_feeding_ms": gpu_out.feeding_ms if gpu_out else None,
            "warp_mn9_hz": gpu_out.mn9_mean_hz if gpu_out else None,
            "warp_distance_cm": gpu_out.distance_cm if gpu_out else None,
            "feeding_ms_diff": (abs(cpu_out.feeding_ms - gpu_out.feeding_ms) if gpu_out else None),
            "distance_cm_diff": (
                round(abs(cpu_out.distance_cm - gpu_out.distance_cm), 5) if gpu_out else None
            ),
        },
    }


def run_batched_embodied_screen(
    pack: ConnectomePack,
    *,
    cell_types: list[str],
    seeds: tuple[int, ...] = (1, 2, 3, 4),
    ticks: int = 250,
    scene: str = "sugar_feeding",
    batch_size: int = 16,
    physics_backend: str = "auto",
    brain_backend: str = "auto",
    verify_top_cpu: bool = True,
) -> dict[str, Any]:
    """Screen cell-type silencings in embodied flies in parallel + CPU float64 confirmation.

    Evaluates ``intact`` + each candidate ``cell_type`` silenced + ``shuffled_control`` across all
    ``seeds``, returning Holm-Bonferroni corrected behavioral effects (``feeding_ms`` or
    ``grooming_ms``) plus exact CPU float64 MuJoCo verification of the strongest hit.
    """
    t0 = time.perf_counter()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    p_back, b_back = _resolve_backends(pack.cells.num_cells, physics_backend, brain_backend)
    n = pack.cells.num_cells

    foods = [FoodSpot(x=0.18, y=0.0, radius=0.45, kind="sugar")] if scene == "sugar_feeding" else []
    poke_ms = ticks if scene == "antennal_grooming" else 0
    metric_attr = "feeding_ms" if scene == "sugar_feeding" else "grooming_ms"

    ct_to_idx: dict[str, I32] = {}
    for ct in cell_types:
        idx = np.asarray([i for i, c in enumerate(pack.cells.cell_type) if c == ct], dtype=np.int32)
        if idx.size > 0:
            ct_to_idx[ct] = idx

    # Build flat list of (condition, seed) jobs for intact + all cell_types
    jobs: list[tuple[str, int]] = [("intact", s) for s in seeds]
    for ct in ct_to_idx:
        for s in seeds:
            jobs.append((ct, s))

    by_cond: dict[str, list[BatchedFlyOutcome]] = {c: [] for c in ["intact", *ct_to_idx]}
    total_fly_s = 0.0
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        sil = np.zeros((len(chunk), n), dtype=np.uint8)
        for ri, (cond, _) in enumerate(chunk):
            if cond != "intact":
                sil[ri, ct_to_idx[cond]] = np.uint8(1)
        runner = BatchedEmbodiedRunner(
            pack,
            csr,
            seeds=[s for _, s in chunk],
            foods=foods,
            poke_ms=poke_ms,
            silenced_mask=sil,
            physics_backend=p_back,
            brain_backend=b_back,
        )
        outs = runner.run(ticks, labels=[c for c, _ in chunk])
        for (cond, _), o in zip(chunk, outs, strict=True):
            by_cond[cond].append(o)
        total_fly_s += len(chunk) * ticks * 1e-3

    # Shuffled-wiring negative control across the same seeds
    shuf_csr = make_shuffled_csr(csr, int(seeds[0]))
    shuf_runner = BatchedEmbodiedRunner(
        pack,
        shuf_csr,
        seeds=list(seeds),
        foods=foods,
        poke_ms=poke_ms,
        physics_backend=p_back,
        brain_backend=b_back,
    )
    shuf_outs = shuf_runner.run(ticks, labels=["shuffled_control"] * len(seeds))
    total_fly_s += len(seeds) * ticks * 1e-3

    base_vals = [float(getattr(o, metric_attr)) for o in by_cond["intact"]]
    base_stats = describe(base_vals)
    base_mean = float(base_stats["mean"])

    raw_rows: list[dict[str, Any]] = []
    p_vals: list[float | None] = []
    for ct, idx in ct_to_idx.items():
        vals = [float(getattr(o, metric_attr)) for o in by_cond[ct]]
        st = describe(vals)
        cmp = compare(base_vals, vals, seed=int(seeds[0])) if len(seeds) >= 2 else None
        pv = float(cmp["permutation"]["p_value"]) if cmp is not None else None
        p_vals.append(pv)
        delta = float(st["mean"]) - base_mean
        supp = (100.0 * (base_mean - float(st["mean"])) / base_mean) if base_mean > 0 else 0.0
        raw_rows.append(
            {
                "cell_type": ct,
                "cell_count": int(idx.size),
                "mean_ms": round(float(st["mean"]), 2),
                "sd_ms": round(float(st["sd"]), 2),
                "delta_ms": round(delta, 2),
                "suppression_pct": round(supp, 1),
                "cliffs_delta": round(float(cmp["cliffs_delta"]), 3) if cmp else None,
                "p_value": round(pv, 4) if pv is not None else None,
            }
        )
    if p_vals:
        holm_adj = holm(p_vals)
        for row, ph in zip(raw_rows, holm_adj, strict=True):
            row["p_value_holm"] = round(float(ph), 4) if ph is not None else None
    raw_rows.sort(key=lambda r: -float(r["suppression_pct"]))

    # Optional exact CPU float64 MuJoCo confirmation of top hit (connectome_plan.md §4 F9!)
    cpu_confirmation: dict[str, Any] | None = None
    if verify_top_cpu and raw_rows:
        top_ct = str(raw_rows[0]["cell_type"])
        w_base = make_embodied_world(
            pack, wiring="reference", seed=int(seeds[0]), backend=b_back, foods=foods
        )
        w_base.arena.poke_remaining_ms = poke_ms
        w_sil = make_embodied_world(
            pack, wiring="reference", seed=int(seeds[0]), backend=b_back, foods=foods
        )
        w_sil.arena.poke_remaining_ms = poke_ms
        w_sil.lif.silenced[ct_to_idx[top_ct]] = np.uint8(1)
        if w_sil.gpu_runner is not None:
            w_sil.gpu_runner.upload_controls(w_sil.lif)
        for _ in range(ticks):
            step_embodied_tick(w_base)
            step_embodied_tick(w_sil)
        base_cpu = w_base.total_feeding_ms if scene == "sugar_feeding" else w_base.total_grooming_ms
        sil_cpu = w_sil.total_feeding_ms if scene == "sugar_feeding" else w_sil.total_grooming_ms
        cpu_confirmation = {
            "top_cell_type": top_ct,
            "seed": int(seeds[0]),
            "cpu_intact_ms": int(base_cpu),
            "cpu_silenced_ms": int(sil_cpu),
            "warp_intact_ms": int(getattr(by_cond["intact"][0], metric_attr)),
            "warp_silenced_ms": int(getattr(by_cond[top_ct][0], metric_attr)),
        }

    wall_s = max(1e-3, time.perf_counter() - t0)
    shuf_vals = [float(getattr(o, metric_attr)) for o in shuf_outs]
    return {
        "dataset_id": pack.dataset_id,
        "num_cells": n,
        "scene": scene,
        "metric": metric_attr,
        "physics_backend": p_back,
        "brain_backend": b_back,
        "batch_size": batch_size,
        "ticks": ticks,
        "seeds": list(seeds),
        "total_evaluations": len(jobs) + len(seeds),
        "wall_seconds": round(wall_s, 3),
        "fly_seconds_per_wall_second": round(total_fly_s / wall_s, 2),
        "baseline": {"mean_ms": round(base_mean, 2), "sd_ms": round(float(base_stats["sd"]), 2)},
        "shuffled_control": {
            "mean_ms": round(float(describe(shuf_vals)["mean"]), 2),
            "sd_ms": round(float(describe(shuf_vals)["sd"]), 2),
        },
        "results": raw_rows,
        "cpu_confirmation": cpu_confirmation,
    }


def run_batched_embodied_search(
    pack: ConnectomePack,
    *,
    generations: int = 4,
    pop_size: int = 8,
    ticks: int = 350,
    seed: int = 1,
    physics_backend: str = "auto",
    brain_backend: str = "auto",
) -> dict[str, Any]:
    """Batched parallel evolutionary search over foraging arousal/steering gains (F9 -> QD/EA)."""
    t0 = time.perf_counter()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    p_back, b_back = _resolve_backends(pack.cells.num_cells, physics_backend, brain_backend)
    foods = [FoodSpot(x=0.85, y=0.35, radius=0.38, kind="sugar")]

    rng = np.random.default_rng(seed)
    mean = np.array([110.0, 1.0], dtype=np.float64)  # [explore_drive_hz, turn_gain]
    std = np.array([45.0, 0.55], dtype=np.float64)
    best_score = -1e9
    best_params = mean.copy()
    best_outcome: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for gen in range(generations):
        cand = np.clip(
            mean[None, :] + std[None, :] * rng.standard_normal((pop_size, 2)),
            [30.0, 0.1],
            [240.0, 2.5],
        )
        cand[0] = mean
        runner = BatchedEmbodiedRunner(
            pack,
            csr,
            seeds=[seed] * pop_size,
            foods=foods,
            explore_drive_hz=cand[:, 0].astype(np.float32),
            turn_gain=cand[:, 1].astype(np.float32),
            physics_backend=p_back,
            brain_backend=b_back,
        )
        outs = runner.run(ticks)
        scores = np.array(
            [
                2.0 * (o.feeding_ms / max(1, ticks)) + max(0.0, 1.2 - max(0.0, o.min_food_dist_cm))
                for o in outs
            ],
            dtype=np.float64,
        )
        order = np.argsort(-scores)
        elite_k = max(2, pop_size // 2)
        mean = cand[order[:elite_k]].mean(axis=0)
        std = np.maximum(np.array([8.0, 0.1]), cand[order[:elite_k]].std(axis=0))
        top_i = int(order[0])
        if float(scores[top_i]) > best_score:
            best_score = float(scores[top_i])
            best_params = cand[top_i].copy()
            best_outcome = {
                "feeding_ms": outs[top_i].feeding_ms,
                "min_food_dist_cm": outs[top_i].min_food_dist_cm,
                "distance_cm": outs[top_i].distance_cm,
            }
        history.append(
            {
                "gen": gen,
                "best_score": round(best_score, 4),
                "gen_best_score": round(float(scores[top_i]), 4),
                "gen_best_min_food_dist_cm": outs[top_i].min_food_dist_cm,
                "gen_best_feeding_ms": outs[top_i].feeding_ms,
            }
        )

    wall_s = max(1e-3, time.perf_counter() - t0)
    return {
        "dataset_id": pack.dataset_id,
        "physics_backend": p_back,
        "brain_backend": b_back,
        "generations": generations,
        "pop_size": pop_size,
        "ticks": ticks,
        "wall_seconds": round(wall_s, 3),
        "fly_seconds_per_wall_second": round((generations * pop_size * ticks * 1e-3) / wall_s, 2),
        "best_score": round(best_score, 4),
        "best_params": {
            "explore_drive_hz": round(float(best_params[0]), 2),
            "turn_gain": round(float(best_params[1]), 3),
        },
        "best_outcome": best_outcome,
        "history": history,
    }
