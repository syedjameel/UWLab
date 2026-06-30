# Linear Gripper vs. Default 2F-85 — Line-by-Line Physics Differences

Reference (working): `Ur5e2f85RobotiqGripperCalibrated/...calibrated.usd` gripper subtree +
`ur5e_robotiq_2f85_gripper.py::ROBOTIQ_2F85`.
Ours: `local/Robots/LinearGripper/linear_gripper.usd` + `ur5e_linear_gripper.py::LINEAR_GRIPPER`.

Extracted with `scratchpad/dump_physics.py` / `dump_bindings.py` (pxr).

## Differences (ranked by likely impact on "grasp not happening")

### 1. Gripping-surface FRICTION  ★ ROOT CAUSE
| | Reference 2F-85 | Ours |
|---|---|---|
| Material on gripping body | `PhysicsMaterial` **bound on the LINK** `left/right_inner_finger` | `JawPhysicsMaterial` **bound on the COLLISION MESH** `.../collisions/JawBox` |
| static/dynamic friction | **100.0** | **0.8** |
| frictionCombineMode | max | max |

With `combineMode=max`, the reference contact friction = max(100, object) = **100** (≈ infinite grip).
Ours = max(0.8, object) = 0.8 → slips. This also explains why sweeping the *slab* friction did
nothing: our jaw capped it at 0.8. **Fix: friction 100 on a material bound to the finger LINK
prims, exactly like the reference.**

### 2. Driver joint maxJointVelocity
| Reference finger_joint | Ours finger_joint (driver) |
|---|---|
| 130 | **0.5** |
Ours throttles the driver to 0.5 (the *mimic* is 130). Reference driver = 130. Normalize to 130.

### 3. Collision geometry on the fingers
- Reference: each finger LINK keeps its real `collisions/mesh_N` (convexHull, enabled=True).
- Ours: finger mesh collision REMOVED, replaced by a synthetic `JawBox` (convexHull). Works
  structurally (collision verified), but is a hand-authored proxy, not the native pad.

### 4. Drive type / gains (USD-baked; overridden by Python actuator at runtime)
| | Reference finger_joint | Ours finger_joint |
|---|---|---|
| joint type | Revolute (angular drive) | Prismatic (linear drive) |
| USD stiffness / damping / maxForce | 0.17 / 0.01 / 16.5 | 200 / 20 / 120 |
| Python actuator stiffness/damping/effort | 17 / 5 / 60 | 50 / 5 / 120 |
Prismatic clamp force ≈ stiffness × (target − contact_pos). At contact fj≈0.049, target 0.068 →
error 0.019 m → 50×0.019 ≈ **0.95 N** clamp (weak). Reference transmits force through the linkage.

### 5. Mechanism (structural — NOT changing)
- Reference: true 4-bar linkage (1 driven + ~7 passive revolute, 2 with excludeFromArticulation=True).
- Ours: 1 driven prismatic + 1 PhysxMimicJoint (gearing −1). Intended design; keep.

### 6. Mass
| | Reference | Ours |
|---|---|---|
| finger/knuckle link mass | not authored (auto from shape+density) | inner_finger 0.15 each |
| base link mass | not authored | robotiq_base_link 0.8 |
| spawn `mass_props` | `MassPropertiesCfg(mass=0.5)` (whole gripper) | none |
Reference normalizes total gripper mass to 0.5 kg at spawn; ours uses baked masses (~1.1 kg) and
sets no mass_props.

### 7. Python actuator (gripper)
| Reference | Ours |
|---|---|
| stiffness 17, damping 5, effort 60 | stiffness 50, damping 5, effort 120 |

## Removal plan (make ours match the reference)
1. **Friction → 100 on finger LINK prims** (diff #1). Highest impact; do first, test locally.
2. **Driver maxJointVelocity 0.5 → 130** (diff #2).
3. If still weak: align actuator gains/mass_props with reference (diffs #4/#6/#7) — set
   `mass_props=mass=0.5`, revisit stiffness so clamp force is a few N.
4. Leave mechanism (diff #5) and the box-proxy collider (diff #3) as-is unless testing shows
   the proxy is the problem.
