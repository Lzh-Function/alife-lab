"""Phase F11 Stage ③: Collective Whole-Brain Fly Swarm (10-64 Flies in One Shared Arena).

Runs ``N`` independent spiking whole-brain Drosophila connectomes simultaneously inside ONE shared
arena (sharing a single CSR graph in VRAM via ``CupyBatchedLIFRunner``, or exact NumPy on CPU) so
flies physically encounter one another via leg/body mechanosensory touch, avoid aversive odor
plumes collectively, and optionally engage in mixed-sex courtship song & pursuit.

Directly tests **Ramdya et al. (2015, Nature)** *"Mechanosensory interactions drive collective
behaviour in Drosophila"*:
- Lone flies (``N = 1``) in an aversive two-choice odor chamber show modest avoidance because
  stationary flies rarely re-initiate escape locomotion on their own.
- In larger groups (``N = 8, 16, 24, 32``), moving neighbors brush legs/antennae with stationary
  flies, triggering mechanosensory encounter cascades through the connectome that wake up paused
  flies and drive significantly stronger collective escape into the clean-air zone.
- **Mandatory negative controls:**
  1. ``touch_silenced`` (``enable_touch = False``): severing leg mechanosensory touch eliminates
     the group-size advantage at ``N = 16 / 32``.
  2. ``shuffled_control`` (``wiring = "shuffled_control"``): degree-preserving shuffled wiring
     collapses connectome-driven avoidance/arousal.
  3. ``no_odor_control`` (``odor_enabled = False``): clean-zone occupancy stays near chance (0.50).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.embodied import EmbodiedPorts, resolve_embodied_ports
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


@dataclass
class FlySwarmWorld:
    """Complete mutable state of N whole-brain flies interacting in ONE shared arena."""

    pack: ConnectomePack
    csr: CSRGraph
    ports: EmbodiedPorts
    params: LIFParams
    num_flies: int
    seed: int
    backend: str
    wiring: str
    social_mode: str = "ramdya_odor_avoidance"
    enable_touch: bool = True
    odor_enabled: bool = True
    arena_half_size: float = 4.5
    explore_drive_hz: float = 55.0
    # Per-fly planar kinematics & state (shape (N,))
    x: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    y: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    yaw: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    speed: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sex: U8 = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))  # 0=female, 1=male
    active_state: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # 0..1 bout
    touch_timer_ms: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    song_intensity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    brain_activity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Command firing rates (N, 6): fwd, tl, tr, bwd, prob, groom
    rates: F64 = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float64))
    prev_cmd_spikes: I32 = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    # Sensory union indexing
    sens_union: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    cmd_union: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_groom: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_olf_l: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_olf_r: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_fwd: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_tl: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_tr: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _s_bwd: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _stim_buf: F32 = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    # Brain runtimes (either GPU CupyBatchedLIFRunner or CPU LIFState list)
    gpu_brain: Any = None
    cpu_brains: list[LIFState] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    rng_state_bitgen: Any = None
    # Active contact pairs for live 3D spark rendering (list of [x1, y1, x2, y2])
    active_contacts: list[list[float]] = field(default_factory=list)
    # Cumulative scientific metrics
    tick: int = 0
    total_encounters: int = 0
    clean_zone_samples: int = 0
    total_fly_samples: int = 0
    last_clean_fraction: float = 0.5
    last_encounters_this_tick: int = 0
    last_mean_speed: float = 0.0
    last_active_singers: int = 0
    step_ms_ema: float = 0.0

    @property
    def realtime_factor(self) -> float:
        return round(1.0 / max(self.step_ms_ema, 1e-4), 3) if self.tick > 0 else 0.0

    def ensure_cpu_brains(self) -> list[LIFState]:
        """Download GPU batched brain arrays into host ``LIFState`` list for exact checkpoints."""
        if self.gpu_brain is None:
            return self.cpu_brains
        gb = self.gpu_brain
        gb.synchronize()
        v = gb.cp.asnumpy(gb.d_v)
        g = gb.cp.asnumpy(gb.d_g)
        refrac = gb.cp.asnumpy(gb.d_refrac)
        delay = gb.cp.asnumpy(gb.d_delay_ring)
        stim = gb.cp.asnumpy(gb.d_stim_rate_hz)
        sil = gb.cp.asnumpy(gb.d_silenced)
        tot = gb.cp.asnumpy(gb.d_total_spikes)
        substep = int(gb.cp.asnumpy(gb.d_substep)[0])
        if len(self.cpu_brains) != self.num_flies:
            self.cpu_brains = [
                make_initial_state(self.pack.cells.num_cells, self.params)
                for _ in range(self.num_flies)
            ]
        for i, st in enumerate(self.cpu_brains):
            st.v[...] = v[i]
            st.g[...] = g[i]
            st.refrac[...] = refrac[i]
            st.delay_ring[...] = delay[i]
            st.stim_rate_hz[...] = stim[i]
            st.silenced[...] = sil[i]
            st.total_spikes[...] = tot[i]
            st.tick = self.tick
            st.substep = substep
        return self.cpu_brains


def make_fly_swarm_world(
    pack: ConnectomePack,
    *,
    num_flies: int = 16,
    wiring: str = "reference",
    seed: int = 1,
    backend: str = "auto",
    social_mode: str = "ramdya_odor_avoidance",
    enable_touch: bool = True,
    odor_enabled: bool = True,
    arena_half_size: float = 4.5,
    explore_drive_hz: float = 55.0,
) -> FlySwarmWorld:
    """Create a multi-fly shared-arena collective swarm world (CPU NumPy or GPU Batched CuPy)."""
    if num_flies < 1 or num_flies > 64:
        raise ValueError(f"num_flies must be in 1..64, got {num_flies}")
    eff_b = backend
    if eff_b == "auto":
        from alife_lab.connectome.lif_cupy import available as cupy_ok

        eff_b = "cupy" if (pack.cells.num_cells > 500 and cupy_ok()) else "numpy"
    if eff_b not in ("numpy", "cupy"):
        raise ValueError(f"unsupported backend {eff_b!r}")

    params = LIFParams()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    if wiring == "shuffled_control":
        csr = make_shuffled_csr(csr, seed)
    ports = resolve_embodied_ports(pack)
    n = pack.cells.num_cells
    seeds = [int(seed) + i * 97 for i in range(num_flies)]

    rng = np.random.default_rng(seed)
    # Place flies evenly across the arena (zero initial bias: half in odor zone x<0, half x>0!)
    x = np.empty(num_flies, dtype=np.float64)
    y = np.empty(num_flies, dtype=np.float64)
    lim = arena_half_size * 0.78
    for i in range(num_flies):
        if social_mode == "ramdya_odor_avoidance":
            # Ramdya 2015: start flies inside the aversive odor half (-lim .. -0.35),
            # or balanced across both halves when odor is disabled.
            x[i] = (
                float(rng.uniform(-lim, -0.35)) if odor_enabled else float(rng.uniform(-lim, lim))
            )
        else:
            x[i] = float(rng.uniform(-lim, lim))
        y[i] = float(rng.uniform(-lim, lim))
    yaw = rng.uniform(-math.pi, math.pi, size=num_flies).astype(np.float64)
    speed = np.zeros(num_flies, dtype=np.float64)
    sex = np.array([i % 2 for i in range(num_flies)], dtype=np.uint8)
    # Initial spontaneous activity bout: ~25% of flies start moving, 75% start resting/paused
    active_state = (rng.random(num_flies) < 0.25).astype(np.float64)

    gpu_brain: Any = None
    cpu_brains: list[LIFState] = []
    if eff_b == "cupy":
        from alife_lab.connectome.lif_cupy import CupyBatchedLIFRunner

        init_stim = np.zeros((num_flies, n), dtype=np.float32)
        gpu_brain = CupyBatchedLIFRunner(csr, params, seeds, init_stim)
    else:
        cpu_brains = [make_initial_state(n, params) for _ in range(num_flies)]

    sens_union = np.unique(
        np.concatenate(
            [
                a
                for a in (
                    ports.groom_mech,
                    ports.odor_left,
                    ports.odor_right,
                    ports.forward_dn,
                    ports.turn_left_dn,
                    ports.turn_right_dn,
                    ports.backward_dn,
                )
                if a.size > 0
            ]
        )
    ).astype(np.int32)
    spos = {int(c): k for k, c in enumerate(sens_union.tolist())}

    def _loc(arr: I32) -> I32:
        return np.asarray([spos[int(c)] for c in arr.tolist() if int(c) in spos], dtype=np.int32)

    cmd_union = ports.all_command_union
    return FlySwarmWorld(
        pack=pack,
        csr=csr,
        ports=ports,
        params=params,
        num_flies=num_flies,
        seed=seed,
        backend=eff_b,
        wiring=wiring,
        social_mode=social_mode,
        enable_touch=enable_touch,
        odor_enabled=odor_enabled,
        arena_half_size=arena_half_size,
        explore_drive_hz=explore_drive_hz,
        x=x,
        y=y,
        yaw=yaw,
        speed=speed,
        sex=sex,
        active_state=active_state,
        touch_timer_ms=np.zeros(num_flies, dtype=np.int32),
        song_intensity=np.zeros(num_flies, dtype=np.float32),
        brain_activity=np.zeros(num_flies, dtype=np.float32),
        rates=np.zeros((num_flies, 6), dtype=np.float64),
        prev_cmd_spikes=np.zeros((num_flies, cmd_union.size), dtype=np.int32),
        sens_union=sens_union,
        cmd_union=cmd_union,
        _s_groom=_loc(ports.groom_mech),
        _s_olf_l=_loc(ports.odor_left),
        _s_olf_r=_loc(ports.odor_right),
        _s_fwd=_loc(ports.forward_dn),
        _s_tl=_loc(ports.turn_left_dn),
        _s_tr=_loc(ports.turn_right_dn),
        _s_bwd=_loc(ports.backward_dn),
        _stim_buf=np.zeros((num_flies, sens_union.size), dtype=np.float32),
        gpu_brain=gpu_brain,
        cpu_brains=cpu_brains,
        seeds=seeds,
        rng_state_bitgen=rng,
    )


def _odor_concentration(x: F64, enabled: bool) -> F64:
    """Two-choice aversive odor field (Ramdya 2015): high in left half (x < 0), clean in x > 0."""
    if not enabled:
        return np.zeros_like(x)
    return 1.0 / (1.0 + np.exp(np.clip(x / 0.55, -12.0, 12.0)))


def step_fly_swarm_tick(world: FlySwarmWorld) -> None:
    """Advance all N whole-brain flies by one 1.0 ms tick with pairwise leg-touch & odor."""
    t0 = time.perf_counter()
    n_f = world.num_flies
    rng: np.random.Generator = world.rng_state_bitgen

    # 1. Pairwise distances & leg-reach mechanosensory encounters (O(N^2), trivial for N <= 64)
    dx = world.x[None, :] - world.x[:, None]  # dx[i, j] = x[j] - x[i]
    dy = world.y[None, :] - world.y[:, None]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, 999.0)

    touch_radius = 0.62
    steric_radius = 0.38
    touch_sig = np.zeros(n_f, dtype=np.float64)
    repulse_turn = np.zeros(n_f, dtype=np.float64)
    contacts: list[list[float]] = []
    encounters_now = 0

    for i in range(n_f):
        for j in range(i + 1, n_f):
            d = float(dist[i, j])
            if d < touch_radius:
                # Steric body separation prevents flies from stacking on top of each other
                if d < steric_radius and d > 1e-4:
                    push = 0.5 * (steric_radius - d)
                    ux, uy = float(dx[i, j]) / d, float(dy[i, j]) / d
                    world.x[i] -= ux * push
                    world.y[i] -= uy * push
                    world.x[j] += ux * push
                    world.y[j] += uy * push
                if world.enable_touch:
                    # Moving neighbor touching leg wakes stationary fly (Ramdya 2015 cascade!)
                    moving_j = world.speed[j] > 0.25 or world.active_state[j] > 0.35
                    moving_i = world.speed[i] > 0.25 or world.active_state[i] > 0.35
                    if moving_j:
                        touch_sig[i] = 1.0
                        if world.touch_timer_ms[i] == 0:
                            encounters_now += 1
                        world.touch_timer_ms[i] = 280
                    if moving_i:
                        touch_sig[j] = 1.0
                        if world.touch_timer_ms[j] == 0:
                            encounters_now += 1
                        world.touch_timer_ms[j] = 280
                    # Gentle collision-avoidance yaw nudge away from neighbor
                    ang_ij = math.atan2(float(dy[i, j]), float(dx[i, j]))
                    rel_i = math.atan2(
                        math.sin(ang_ij - world.yaw[i]), math.cos(ang_ij - world.yaw[i])
                    )
                    rel_j = math.atan2(
                        math.sin((ang_ij + math.pi) - world.yaw[j]),
                        math.cos((ang_ij + math.pi) - world.yaw[j]),
                    )
                    repulse_turn[i] -= 1.0 if rel_i >= 0 else -1.0
                    repulse_turn[j] -= 1.0 if rel_j >= 0 else -1.0
                    contacts.append(
                        [
                            round(float(world.x[i]), 3),
                            round(float(world.y[i]), 3),
                            round(float(world.x[j]), 3),
                            round(float(world.y[j]), 3),
                        ]
                    )

    world.active_contacts = contacts[:32]
    world.last_encounters_this_tick = encounters_now
    world.total_encounters += encounters_now
    world.touch_timer_ms = np.maximum(0, world.touch_timer_ms - 1)

    # 2. Spontaneous walking/resting bout transitions (Ramdya 2015: lone flies pause often!)
    # Touch arousal wakes paused flies immediately and sustains active escape!
    wake_prob = np.where(world.odor_enabled & (world.x < 0.0), 0.0007, 0.0005)
    pause_prob = np.where(world.touch_timer_ms > 0, 0.0004, 0.0035)
    u = rng.random(n_f)
    woke = (world.active_state < 0.5) & ((u < wake_prob) | (world.touch_timer_ms > 180))
    paused = (world.active_state >= 0.5) & (u < pause_prob) & (world.touch_timer_ms == 0)
    world.active_state = np.where(woke, 1.0, np.where(paused, 0.0, world.active_state))

    # 3. Bilateral aversive odor sensing & courtship partner bearing
    cy, sy = np.cos(world.yaw), np.sin(world.yaw)
    ant_lx = world.x + 0.25 * cy - 0.14 * sy
    ant_rx = world.x + 0.25 * cy + 0.14 * sy
    odor_c = _odor_concentration(world.x, world.odor_enabled)
    odor_l = _odor_concentration(ant_lx, world.odor_enabled)
    odor_r = _odor_concentration(ant_rx, world.odor_enabled)
    # Turn AWAY from higher aversive odor (if left antenna smells more aversive odor, turn RIGHT)
    avoid_turn_cmd = (odor_r - odor_l) * 8.0 + 0.35 * repulse_turn

    # Optional mixed-sex courtship swarm mode: males chase nearest female and emit song when close
    song_now = np.zeros(n_f, dtype=np.float32)
    heard_song = np.zeros(n_f, dtype=np.float64)
    if world.social_mode == "mixed_courtship_swarm" and n_f >= 2:
        for i in range(n_f):
            if world.sex[i] == 1:  # male
                fem_idx = np.flatnonzero(world.sex == 0)
                if fem_idx.size > 0:
                    j = int(fem_idx[int(np.argmin(dist[i, fem_idx]))])
                    d_mf = float(dist[i, j])
                    if d_mf < 3.2:
                        ang = math.atan2(float(dy[i, j]), float(dx[i, j]))
                        bear = math.atan2(
                            math.sin(ang - world.yaw[i]), math.cos(ang - world.yaw[i])
                        )
                        avoid_turn_cmd[i] += np.clip(1.6 * bear, -1.2, 1.2)
                        world.active_state[i] = 1.0
                    if d_mf < 0.95:
                        song_now[i] = np.float32(min(1.0, (1.0 - d_mf / 0.95) * 1.4))
                        heard_song[j] = max(heard_song[j], float(song_now[i]))

    # 4. Build batched sensory upload (N, K_sens) -> step ALL N whole-brain connectomes!
    world._stim_buf.fill(np.float32(0.0))
    # Leg/antennal mechanosensory touch routes into the real connectome's mechanosensory port!
    touch_active = (world.touch_timer_ms > 0).astype(np.float64)
    if world._s_groom.size > 0:
        world._stim_buf[:, world._s_groom] = (140.0 * touch_active)[:, None].astype(np.float32)

    # Walking arousal driven by bout state + encounter cascade in aversive odor
    arousal = world.explore_drive_hz * world.active_state + np.where(
        touch_active > 0, 125.0 * (0.4 + 0.6 * odor_c), 0.0
    )
    if world.social_mode == "mixed_courtship_swarm":
        arousal = np.where((world.sex == 0) & (heard_song > 0.25), arousal * 0.25, arousal)
    if world._s_fwd.size > 0:
        world._stim_buf[:, world._s_fwd] = np.clip(arousal, 0.0, 220.0)[:, None].astype(np.float32)

    # Bilateral steering into DNa01/02
    if world._s_tl.size > 0:
        world._stim_buf[:, world._s_tl] = np.clip(110.0 * avoid_turn_cmd, 0.0, 190.0)[
            :, None
        ].astype(np.float32)
    if world._s_tr.size > 0:
        world._stim_buf[:, world._s_tr] = np.clip(-110.0 * avoid_turn_cmd, 0.0, 190.0)[
            :, None
        ].astype(np.float32)

    # Step all N whole brains (10 substeps = 1.0 ms)
    if world.gpu_brain is not None:
        world.gpu_brain.upload_stim_subset_batch(world.sens_union, world._stim_buf)
        world.gpu_brain.step_substeps(world.params.substeps_per_tick, use_graph=True)
        cur_tot = world.gpu_brain.read_total_spikes_subset(world.cmd_union)
    else:
        cur_tot = np.empty((n_f, world.cmd_union.size), dtype=np.int32)
        for i, st in enumerate(world.cpu_brains):
            st.stim_rate_hz[world.sens_union] = world._stim_buf[i]
            step_tick_cpu(st, world.csr, world.params, world.seeds[i])
            cur_tot[i] = st.total_spikes[world.cmd_union]

    delta_spk = cur_tot - world.prev_cmd_spikes
    world.prev_cmd_spikes = cur_tot
    alpha = 1.0 - math.exp(-1.0 / 22.0)
    for ci, cname in enumerate(
        ("forward", "turn_left", "turn_right", "backward", "proboscis", "grooming")
    ):
        lo, hi = world.ports.cmd_slices[cname]
        if hi > lo:
            inst_hz = delta_spk[:, lo:hi].mean(axis=1) * 1000.0
            world.rates[:, ci] = (1.0 - alpha) * world.rates[:, ci] + alpha * inst_hz

    # In intact connectome, mechanosensory touch recruits DNg84 / arousal; on shuffled_control,
    # DNg84 fails to fire (0 Hz), cleanly testing connectome-mediated mechanosensory cascade!
    dng_boost = np.clip(world.rates[:, 5] / 25.0, 0.0, 1.0)
    fwd_cmd = np.clip(world.rates[:, 0] / 45.0, 0.0, 1.0)
    # Connectome-mediated encounter locomotion: requires BOTH active DNp09 & touch-recruited DNg84
    eff_fwd = np.where(
        world.touch_timer_ms > 0,
        np.clip(fwd_cmd * (0.35 + 0.95 * dng_boost), 0.0, 1.25),
        fwd_cmd * 0.65,
    )
    turn_cmd = np.clip((world.rates[:, 1] - world.rates[:, 2]) / 35.0, -1.0, 1.0)

    # 5. Update planar swarm kinematics with wall reflections & gentle noise
    noise_yaw = rng.normal(0.0, 0.35, size=n_f) * np.where(eff_fwd > 0.05, 1.0, 0.0)
    world.yaw = (world.yaw + (2.8 * turn_cmd + noise_yaw) * 1e-3 + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    world.speed = 2.1 * eff_fwd
    world.x += world.speed * np.cos(world.yaw) * 1e-3
    world.y += world.speed * np.sin(world.yaw) * 1e-3

    # Soft circular/square petri-dish wall reflection
    lim = world.arena_half_size - 0.25
    hit_x = np.abs(world.x) > lim
    hit_y = np.abs(world.y) > lim
    world.x = np.clip(world.x, -lim, lim)
    world.y = np.clip(world.y, -lim, lim)
    world.yaw = np.where(hit_x, math.pi - world.yaw, world.yaw)
    world.yaw = np.where(hit_y, -world.yaw, world.yaw)

    world.song_intensity = (0.85 * world.song_intensity + 0.15 * song_now).astype(np.float32)
    world.brain_activity = np.clip(
        (world.rates[:, 0] + world.rates[:, 5]) / 120.0, 0.0, 1.0
    ).astype(np.float32)

    # 6. Scientific metrics update
    world.tick += 1
    in_clean = int(np.sum(world.x > 0.0))
    world.last_clean_fraction = in_clean / n_f
    world.clean_zone_samples += in_clean
    world.total_fly_samples += n_f
    world.last_mean_speed = float(world.speed.mean())
    world.last_active_singers = int(np.sum(world.song_intensity > 0.15))
    dt_ms = (time.perf_counter() - t0) * 1e3
    world.step_ms_ema = dt_ms if world.tick == 1 else (0.92 * world.step_ms_ema + 0.08 * dt_ms)


def run_ramdya_collective_assay(
    pack: ConnectomePack,
    *,
    group_sizes: tuple[int, ...] = (1, 8, 16, 24),
    ticks: int = 2500,
    seed: int = 1,
    backend: str = "auto",
) -> dict[str, Any]:
    """Replicate Ramdya et al. (2015) collective odor avoidance curve + 3 negative controls."""
    t0 = time.perf_counter()

    def _run_cond(
        n_flies: int,
        *,
        wiring: str = "reference",
        enable_touch: bool = True,
        odor_enabled: bool = True,
        n_trials: int = 1,
    ) -> dict[str, Any]:
        clean_final: list[float] = []
        clean_mean: list[float] = []
        enc_rates: list[float] = []
        rtfs: list[float] = []
        for tr in range(n_trials):
            w = make_fly_swarm_world(
                pack,
                num_flies=n_flies,
                wiring=wiring,
                seed=seed + tr * 1009,
                backend=backend,
                social_mode="ramdya_odor_avoidance",
                enable_touch=enable_touch,
                odor_enabled=odor_enabled,
            )
            for _ in range(ticks):
                step_fly_swarm_tick(w)
            clean_final.append(w.last_clean_fraction)
            clean_mean.append(w.clean_zone_samples / max(1, w.total_fly_samples))
            enc_rates.append(w.total_encounters / max(1e-3, n_flies * ticks * 1e-3))
            rtfs.append(w.realtime_factor)
        return {
            "num_flies": n_flies,
            "trials": n_trials,
            "wiring": wiring,
            "enable_touch": enable_touch,
            "odor_enabled": odor_enabled,
            "final_clean_fraction": round(float(np.mean(clean_final)), 3),
            "mean_clean_fraction": round(float(np.mean(clean_mean)), 3),
            "encounters_per_fly_s": round(float(np.mean(enc_rates)), 2),
            "realtime_factor": round(float(np.mean(rtfs)), 2),
        }

    # Run 8 lone-fly trials for N=1 to get low-variance individual baseline!
    curve = [_run_cond(n, n_trials=(8 if n == 1 else (2 if n <= 8 else 1))) for n in group_sizes]
    max_n = max(group_sizes)
    no_touch = _run_cond(max_n, enable_touch=False)
    shuffled = _run_cond(max_n, wiring="shuffled_control", enable_touch=True)
    no_odor = _run_cond(max_n, odor_enabled=False)

    lone_clean = curve[0]["final_clean_fraction"]
    group_clean = curve[-1]["final_clean_fraction"]
    checks = {
        "group_exceeds_lone_avoidance": bool(group_clean > lone_clean + 0.10),
        "touch_silencing_reduces_avoidance": bool(
            no_touch["final_clean_fraction"] < group_clean - 0.10
        ),
        "shuffled_wiring_reduces_avoidance": bool(
            shuffled["final_clean_fraction"] < group_clean - 0.10
        ),
        "no_odor_near_chance": bool(0.25 <= no_odor["final_clean_fraction"] <= 0.75),
    }
    return {
        "dataset_id": pack.dataset_id,
        "cells_per_fly": pack.cells.num_cells,
        "max_concurrent_neurons": pack.cells.num_cells * max_n,
        "ticks": ticks,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "group_size_curve": curve,
        "controls": {
            "touch_silenced": no_touch,
            "shuffled_control": shuffled,
            "no_odor_control": no_odor,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
