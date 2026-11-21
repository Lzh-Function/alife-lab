# Substrate contract — `creature`

- State: MuJoCo `MjData` (checkpoint: `mj_getState(mjSTATE_INTEGRATION)`), tick, start
  position, cumulative motor work, recent torso positions (for speed).
- Mode: exploratory only. Clock: tick = `substeps` physics steps. Backend: `numpy` (MuJoCo C).
- Spec: `dynamics` timestep, substeps, gear, friction, density, joint_stiffness,
  `body` (torso_radius, legs [{angle_deg, upper, lower, radius}] ≤ 8), `controller`
  (kind cpg|off, frequency_hz, amplitudes, phases: 2 per leg, ordered hip, knee);
  `initialization.height`.
- Interventions: `push {vx, vy, vz}` (m/s, |·| ≤ 10) added to the torso velocity.
- Inspection: torso position, velocity, uprightness, mass; per leg hip/knee angles, controls
  and foot contact.
- Metrics: displacement, forward, speed, height, upright, energy, time.
- Display codec `scene3d_f32`: header `geoms` [{type, size, rgba}], `torso`; payload float32
  per geom position (3) + rotation matrix (9).
