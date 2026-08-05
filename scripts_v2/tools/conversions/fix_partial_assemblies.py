# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Move ungraspable partial-assembly poses into the insertion corridor (CPU only, no Isaac).

THE DEFECT. ``ObjectPartiallyAssembledEEGrasped`` places the board at a partial-assembly pose,
then puts the EE on it from the grasp dataset. That needs the jaws to reach around the board --
but the jig window is 102 mm wide (+-51), the board is 100 mm, and the jaws close at +-50 with
their pads extending OUTWARD past that. So for any pose that puts the board INSIDE the window the
grasp is geometrically impossible: the event leaves the jaws open, the board drops, and it seats.
``check_reset_state_success`` validates stability, not attachment, so that "stable, assembled,
ungripped" state is then accepted.

Measured on a 64-state C4 sample:
    jaws OPEN 31/64 (48%)   seated 44/64 (69%)   useful (held, not seated) 3/64 (5%)
with source poses at p25 rel-z 9.76 mm against a rim at 20.8 mm -- ~40-45% inside the window,
matching the 48% failure rate.

THE FIX. Do not discard those poses -- MOVE them, so the whole sampled distribution is kept:
  * lift any board centre below the rim up into the corridor just above it, and
  * push poses that sit squarely over the window opening out to a lateral offset, so the board
    overhangs the rim. Then even a failed grasp lands it ON the rim instead of seated, and the
    policy still has the final alignment left to do rather than starting solved.

Legitimate because C4 states are HELD by the gripper -- the pose has to be reachable, not stably
restable. Deviation, physically forced: a drop-in assembly has no graspable partial state.

    python scripts_v2/tools/conversions/fix_partial_assemblies.py --in-place \
      --input ./Datasets_realpcb_jig/OmniReset/Resets/JigEnclosure__RealPcb/partial_assemblies.pt
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import torch

# assembly frame: origin at the pair's bbox centre; jig top face +20.8 mm;
# seated board centre -5.7 mm (assembled_offset -7.2 + board half-thickness 1.5)
RIM_Z_MM = 20.8
SEATED_Z_MM = -5.7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--in-place", action="store_true", help="overwrite input (keeps a .bak)")
    ap.add_argument("--min-rel-z-mm", type=float, default=RIM_Z_MM + 4.0,
                    help="lift board centres below this into the corridor (default 24.8 = rim + 4)")
    ap.add_argument("--lateral-mm", type=float, default=18.0,
                    help="minimum in-plane offset from the window centre, so the board overhangs "
                         "the rim and a failed grasp cannot drop it into the seat")
    ap.add_argument("--lateral-span-mm", type=float, default=12.0,
                    help="width of the random band above --lateral-mm, preserving spread")
    ap.add_argument("--max-tilt-deg", type=float, default=5.0,
                    help="flatten board orientations tilted beyond this back to near-upright, "
                         "keeping yaw. assembly_sampling_event runs with friction ZERO and random "
                         "forces, which a heavy jig resists but a 3 mm board does not -- it simply "
                         "tips over (measured: median tilt 26.1 deg, p90 78.4, max 83.8). A tilted "
                         "board tilts the grasp frame with it, so orient_down fails and the jaws "
                         "cannot hold it: the board tumbles out (ang|v| median 5.34, max 25 rad/s) "
                         "and the not_far check rejects the state. Only 1-11 of 32 survived.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = torch.load(args.input, map_location="cpu", weights_only=False)
    pos = d["relative_position"].clone().numpy().astype(np.float64)
    n = len(pos)
    z0 = pos[:, 2] * 1000.0
    r0 = np.linalg.norm(pos[:, :2], axis=1) * 1000.0

    # 1. lift below-rim poses into the corridor (keeps their xy)
    low = z0 < args.min_rel_z_mm
    pos[low, 2] = args.min_rel_z_mm / 1000.0

    # 2. push well-centred poses out laterally so the board overhangs the rim
    # The rim ramps are a FUNNEL (built deliberately, for capture), so a board released even a
    # few mm off-centre is guided back in and seats. To land ON the rim instead, the offset must
    # clear the funnel mouth: (154-140)/2 = 7 mm in x, (118.5-100)/2 = 9.25 mm in y. Push to a
    # RANDOM radius in [lateral, lateral+span] rather than a fixed one -- clamping every pose to
    # a single value collapsed 103/104 onto one ring and threw away the sampled spread.
    centred = r0 < args.lateral_mm
    if centred.any():
        k = int(centred.sum())
        ang = rng.uniform(0.0, 2.0 * np.pi, k)
        rad = rng.uniform(args.lateral_mm, args.lateral_mm + args.lateral_span_mm, k) / 1000.0
        pos[centred, 0] = rad * np.cos(ang)
        pos[centred, 1] = rad * np.sin(ang)

    # 3. flatten tumbled orientations to near-upright, PRESERVING YAW.
    # Legitimate for the same reason as the position fix: C4 states are HELD, so the pose has to
    # be graspable, not physically settled. The underlying flaw is in the generator (zero friction
    # tumbles thin parts) -- left alone here because it is shared with the jig and pcb tasks.
    quat = d["relative_orientation"].clone().numpy().astype(np.float64)
    w, x, y, zq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    tilt0 = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))
    tilted = tilt0 > args.max_tilt_deg
    if tilted.any():
        yaw = np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        k = int(tilted.sum())
        # yaw-only quaternion, plus a small random roll/pitch so the set is not perfectly flat
        rp = np.radians(rng.uniform(-args.max_tilt_deg, args.max_tilt_deg, (k, 2)))
        cy, sy = np.cos(yaw[tilted] / 2), np.sin(yaw[tilted] / 2)
        cr, sr = np.cos(rp[:, 0] / 2), np.sin(rp[:, 0] / 2)
        cp, sp = np.cos(rp[:, 1] / 2), np.sin(rp[:, 1] / 2)
        quat[tilted, 0] = cr * cp * cy + sr * sp * sy
        quat[tilted, 1] = sr * cp * cy - cr * sp * sy
        quat[tilted, 2] = cr * sp * cy + sr * cp * sy
        quat[tilted, 3] = cr * cp * sy - sr * sp * cy
    w2, x2, y2 = quat[:, 0], quat[:, 1], quat[:, 2]
    tilt1 = np.degrees(np.arccos(np.clip(1 - 2 * (x2 * x2 + y2 * y2), -1.0, 1.0)))

    z1 = pos[:, 2] * 1000.0
    r1 = np.linalg.norm(pos[:, :2], axis=1) * 1000.0
    print(f"[fix] {args.input}   n={n} (none discarded)")
    print(f"  rel z : {z0.min():7.2f}..{z0.max():7.2f}  ->  {z1.min():7.2f}..{z1.max():7.2f} mm "
          f"| lifted {int(low.sum())}")
    print(f"  radius: {r0.min():7.2f}..{r0.max():7.2f}  ->  {r1.min():7.2f}..{r1.max():7.2f} mm "
          f"| pushed out {int(centred.sum())}")
    print(f"  seated-equivalent poses: {int((np.abs(z0 - SEATED_Z_MM) < 5).sum())} -> "
          f"{int((np.abs(z1 - SEATED_Z_MM) < 5).sum())}")
    print(f"  below rim ({RIM_Z_MM} mm, ungraspable): {int((z0 < RIM_Z_MM).sum())} -> "
          f"{int((z1 < RIM_Z_MM).sum())}")
    print(f"  tilt  : median {np.median(tilt0):6.1f} -> {np.median(tilt1):5.1f} deg, "
          f"max {tilt0.max():6.1f} -> {tilt1.max():5.1f} | flattened {int(tilted.sum())}")

    out = dict(d)
    out["relative_position"] = torch.as_tensor(pos, dtype=d["relative_position"].dtype)
    out["relative_orientation"] = torch.as_tensor(quat, dtype=d["relative_orientation"].dtype)
    if "relative_pose" in out and torch.is_tensor(out["relative_pose"]) and len(out["relative_pose"]) == n:
        rp = out["relative_pose"].clone()
        rp[:, :3] = out["relative_position"]
        rp[:, 3:7] = out["relative_orientation"]
        out["relative_pose"] = rp

    dst = args.input if args.in_place else (args.output or args.input + ".fixed.pt")
    if args.in_place and not os.path.exists(args.input + ".bak"):
        shutil.copy2(args.input, args.input + ".bak")
        print(f"  backup -> {args.input}.bak")
    torch.save(out, dst)
    print(f"  wrote  -> {dst}")


if __name__ == "__main__":
    main()
