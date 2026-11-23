"""Phase F8 Level 1 reference tests: real-time embodied Drosophila (Brain <-> Body).

Pure-NumPy spec validation, synthetic embodied connectome pack, and MJCF generation tests run
unconditionally in CPU-only CI. MuJoCo closed-loop physics, checkpoint round-trip, live
intervention, and CLI verification tests are guarded with ``@needs_mujoco``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.embodied import (
    EMBODIED_LEVEL1_FRAMING,
    make_synthetic_fly_embodied_pack,
    resolve_embodied_ports,
    run_embodied_verification,
)
from alife_lab.connectome.fly_body import build_fly_mjcf
from alife_lab.core.spec import ExperimentSpec, load_experiment_spec
from alife_lab.substrates.connectome.embodied_adapter import FlyEmbodiedAdapter

ROOT = Path(__file__).resolve().parents[2]
HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(
    not HAS_MUJOCO, reason="MuJoCo not installed (uv sync --extra embodied)"
)


def test_fly_embodied_spec_and_synthetic_pack_cpu_only() -> None:
    raw_ref = {
        "experiment": {
            "name": "embodied_ref_reject",
            "mode": "reference",
            "substrate": "fly_embodied",
            "seed": 1,
            "dataset_id": "synthetic_fly_embodied",
        }
    }
    with pytest.raises(Exception, match="exploratory"):
        ExperimentSpec.model_validate(raw_ref)

    spec = load_experiment_spec(ROOT / "configs/scenarios/fly_embodied_tiny.yaml")
    assert spec.experiment.substrate == "fly_embodied"
    assert "hand-crafted" in EMBODIED_LEVEL1_FRAMING.lower()

    pack = make_synthetic_fly_embodied_pack()
    assert pack.cells.num_cells == 24
    ports = resolve_embodied_ports(pack)
    assert ports.sugar_grn.size == 4
    assert ports.bitter_grn.size == 2
    assert ports.groom_mech.size == 2
    assert ports.proboscis_mn.size == 2
    assert ports.grooming_dn.size == 2
    assert ports.forward_dn.size == 1

    mjcf = build_fly_mjcf()
    assert 'model="alife_embodied_fly_l1"' in mjcf
    assert 'name="act_proboscis"' in mjcf


@needs_mujoco
def test_proboscis_never_pierces_floor() -> None:
    import mujoco

    from alife_lab.connectome.fly_body import make_fly_body

    b = make_fly_body()
    m, d = b.model, b.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_proboscis")
    half_len = float(m.geom_size[gid][1])
    rad = float(m.geom_size[gid][0])
    for ctrl_val in (0.0, 0.45, 0.95):
        d.ctrl[18] = ctrl_val
        for _ in range(150):
            mujoco.mj_step(m, d)
        axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
        p1 = d.geom_xpos[gid] + axis * half_len
        p2 = d.geom_xpos[gid] - axis * half_len
        lowest_z = min(float(p1[2]), float(p2[2])) - rad
        assert lowest_z > 0.02, f"proboscis pierced floor at ctrl={ctrl_val}: z={lowest_z:.4f}"


@needs_mujoco
def test_fly_stays_upright_during_walking_and_long_grooming() -> None:
    import math

    from alife_lab.connectome.embodied import make_embodied_world, step_embodied_tick

    pack = make_synthetic_fly_embodied_pack()
    w = make_embodied_world(pack, wiring="reference", seed=1, backend="numpy", foods=[])
    # 500 ms free walking + 600 ms continuous grooming: fly must NEVER rear up or flip!
    for t in range(1100):
        if t == 500:
            w.arena.poke_remaining_ms = 600
        step_embodied_tick(w)
        q = w.body.data.qpos
        qw, qx, qy, qz = float(q[3]), float(q[4]), float(q[5]), float(q[6])
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        pitch_deg = abs(math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))) * 180.0 / math.pi)
        assert upright > 0.90, f"fly tipped over at tick {t}: upright={upright:.3f}"
        assert pitch_deg < 20.0, f"fly reared up at tick {t}: pitch={pitch_deg:.1f}deg"


@needs_mujoco
def test_embodied_verification_all_six_gates_pass() -> None:
    pack = make_synthetic_fly_embodied_pack()
    rep = run_embodied_verification(pack, backend="numpy", seed=1, ticks_per_scene=200)
    assert rep["all_checks_passed"] is True, rep["checks"]
    assert rep["scenes"]["sugar_feeding"]["feeding_ms"] > 100
    assert rep["scenes"]["bitter_suppression"]["feeding_ms"] == 0
    assert rep["scenes"]["antennal_grooming"]["grooming_ms"] > 100
    assert rep["scenes"]["silenced_mn9"]["feeding_ms"] == 0


@needs_mujoco
def test_embodied_adapter_interventions_checkpoint_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/fly_embodied_tiny.yaml")
    ad = FlyEmbodiedAdapter()
    ad.set_backend("numpy")
    w = ad.initialize(spec)
    for _ in range(120):
        ad.step(w, spec)

    m = ad.metrics(w, spec)
    assert m["feeding_ms"] > 30
    assert m["realtime_factor"] > 0.0

    # Render packet contains both MuJoCo geom transforms and neural fired indices.
    hdr, raw = ad.render(w)
    assert hdr["codec"] == "fly_embodied_v1"
    assert hdr["geom_bytes"] == hdr["geom_count"] * 12 * 4
    assert len(raw) == hdr["geom_bytes"] + hdr["fired_count"] * 4

    # Live interventions: poke antenna triggers grooming; silence MN9 blocks feeding.
    ad.apply_intervention(w, spec, "poke_antenna", {"duration_ms": 200})
    for _ in range(120):
        ad.step(w, spec)
    assert w.total_grooming_ms > 30

    info = ad.inspect(w, spec, {"body_part": "proboscis"})
    assert info["substrate"] == "fly_embodied"
    assert info["body_part_link"]["cell_types"] == ["CB0701"]

    # Exact deterministic checkpoint save -> restore -> step equivalence (both brain AND body!).
    ckpt = ad.serialize_state(w)
    w2 = ad.deserialize_state(ckpt, spec)
    assert ad.state_distance(w, w2)["exact_equal"] is True
    for _ in range(30):
        ad.step(w, spec)
        ad.step(w2, spec)
    assert ad.state_distance(w, w2)["exact_equal"] is True

    # CLI JSON smoke test.
    rc = cli_main(
        [
            "connectome",
            "embody",
            "synthetic_fly_embodied",
            "--backend",
            "numpy",
            "--ticks",
            "180",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_checks_passed"] is True


@needs_mujoco
def test_render_body_poses_are_horizontal() -> None:
    """CT-scanned anatomical STL meshes must attach to unrotated MuJoCo body frames (+Z up, +X fwd),
    not capsule geoms (whose local +Z aligns with fromto and pitches the thorax 90 deg vertical).
    """
    spec = load_experiment_spec(ROOT / "configs/scenarios/fly_embodied_tiny.yaml")
    adapter = FlyEmbodiedAdapter()
    state = adapter.initialize(spec)
    for _ in range(30):
        state = adapter.step(state, spec)
    header, _ = adapter.render(state)
    bp = header["body_poses"]
    for name in ("thorax", "head", "proboscis"):
        assert name in bp
        pos = bp[name]["pos"]
        mat = bp[name]["mat"]
        assert pos[2] > 0.14, f"{name} sank too low: z={pos[2]}"
        # Row-major 3x3: mat[8] = R[2,2] (dorsal +Z aligned with world +Z), mat[0] = R[0,0]
        assert mat[8] > 0.90, f"{name} body frame tilted vertically: R22={mat[8]}"
        assert mat[0] > 0.85, f"{name} body frame unexpectedly rotated: R00={mat[0]}"


def test_level2_synthetic_cns_pack_cpu_only() -> None:
    """CPU-only verification of the 72-neuron synthetic Brain+VNC pack (36 leg MNs + 6 proprio)."""
    from alife_lab.connectome.embodied import make_synthetic_cns_embodied_pack

    pack = make_synthetic_cns_embodied_pack()
    assert pack.cells.num_cells == 72
    mn_ports = [k for k in pack.io_ports if k.startswith("mn_")]
    pr_ports = [k for k in pack.io_ports if k.startswith("proprio_")]
    assert len(mn_ports) == 36
    assert len(pr_ports) == 6
    ports = resolve_embodied_ports(pack)
    assert ports.has_vnc_ports is True
    assert len(ports.proprio_ports) == 6
    assert len(ports.leg_mn_slices) == 36


@needs_mujoco
def test_level2_vnc_direct_drive_controls_and_checkpoint() -> None:
    """Level 2 VNC direct drive (NO hand-crafted CPG): passes all 4 checks + bitwise checkpoint."""
    from alife_lab.connectome.embodied import (
        make_synthetic_cns_embodied_pack,
        run_embodied_level2_verification,
    )

    pack = make_synthetic_cns_embodied_pack()
    rep = run_embodied_level2_verification(pack, backend="numpy", seed=7, ticks_per_scene=250)
    assert rep["all_checks_passed"] is True
    assert rep["scenes"]["vnc_intact"]["vnc_mn_peak_hz"] >= 1.0

    # Bitwise Brain+VNC+MuJoCo checkpoint round-trip on Level 2 preset
    spec = load_experiment_spec(ROOT / "configs/scenarios/fly_embodied_vnc_tiny.yaml")
    adapter = FlyEmbodiedAdapter()
    s1 = adapter.initialize(spec)
    for _ in range(40):
        s1 = adapter.step(s1, spec)
    payload = adapter.serialize_state(s1)
    s2 = adapter.deserialize_state(payload, spec)
    for _ in range(30):
        s1 = adapter.step(s1, spec)
        s2 = adapter.step(s2, spec)
    dist = adapter.state_distance(s1, s2)
    assert dist["exact_equal"] is True
    assert dist["qpos_max_abs_diff"] == 0.0
    assert dist["qvel_max_abs_diff"] == 0.0

    hdr, _ = adapter.render(s1)
    assert hdr["control_mode"] == "level2_vnc"
    assert hdr["has_vnc_ports"] is True
    assert hdr["leg_mn_hz"] is not None and len(hdr["leg_mn_hz"]) == 6


@needs_mujoco
def test_level2_rejects_brain_only_pack_and_cli_works(capsys: pytest.CaptureFixture[str]) -> None:
    """Brain-only packs (without VNC leg MNs) must be rejected cleanly in Level 2 mode."""
    from alife_lab.connectome.embodied import make_embodied_world

    brain_only = make_synthetic_fly_embodied_pack()
    with pytest.raises(ValueError, match="VNC"):
        make_embodied_world(brain_only, control_mode="level2_vnc")

    rc = cli_main(["connectome", "embody-l2", "synthetic_cns_embodied", "--ticks", "120", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_checks_passed"] is True
