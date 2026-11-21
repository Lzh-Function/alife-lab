# Substrate contract — `creature`

- State: MuJoCo `MjData` (checkpoint: `mj_getState(mjSTATE_INTEGRATION)`), tick, start
  position, cumulative motor work, recent torso positions (for speed), path length, and the
  reflex's sensed orientation and foot contacts (one-step delay).
- Mode: exploratory only. Clock: tick = `substeps` physics steps. Backend: `numpy` (MuJoCo C).
- Spec: `dynamics` timestep, substeps, gear, friction, density, joint_stiffness,
  `body` (torso_radius, legs [{angle_deg, upper, lower, radius}] ≤ 8), `controller`
  (kind cpg|off, frequency_hz, amplitudes, phases: 2 per leg, ordered hip, knee), optional
  `reflex` (tilt_gain, contact_gain); `initialization.height`; optional `terrain` (kind
  rough, amplitude, wavelength, half_size, resolution, flat_radius; heights from the seed).
- Interventions: `push {vx, vy, vz}` (m/s, |·| ≤ 10) added to the torso velocity.
- Inspection: torso position, velocity, uprightness, mass, tilt, path length, reflex, terrain
  parameters; per leg hip/knee angles, controls and foot contact. `scene=true` returns the
  terrain heights for display (base64 uint16 of 0..amplitude, ≤ 241² samples, row = y).
- Metrics: displacement, forward, speed, height, upright, energy, time.
- Display codec `scene3d_f32`: header `geoms` [{name, type, size, rgba}], `torso`, `start`,
  `contacts`, `reflex`, `tilt_deg`, `path_length`, `time`, `trail` [[x, y]]; payload float32
  per geom position (3) + rotation matrix (9). The `hfield` geom is drawn from `scene` data.
