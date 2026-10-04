#!/usr/bin/env bash
# Acceptance test for a trained checkpoint -- CONTACT FIRST, success rate last.
#
# WHY THIS EXISTS AND WHY IT IS SHAPED THIS WAY. Every wrong conclusion in this project came from
# scoring a policy on a proxy:
#
#   * `in_hand` is |d - d0| < 0.02, the CHANGE in cube-to-fingertip distance. It reported 1.000 --
#     a flawless grasp -- for a cube hanging motionless 56 mm from wide-open fingers. Measured
#     fingertip force through those same rollouts was 0.0000 N.
#   * C4 `task_0_success_rate` ~0.59 was the RESET DISTRIBUTION: the cube starts seated within
#     tolerance, median displacement over 150 steps is 0.00000 m, and success went 39/64 at step 0
#     to 38/64 at step 149. The policy never touched the cube and never moved it.
#
# So this script refuses to report a success rate until it has answered, from the scene's own
# ContactSensorCfg (`object_hand_s`, filtered to the five fingertips):
#
#   1. does the hand exert measurable force on the cube?
#   2. is the two-finger opposition gate (thumb rl_dg_1_tip AND index rl_dg_2_tip, both above
#      GRASP_CONTACT_THRESHOLD_N = 0.2) open in more than 0/64 environments?
#   3. does the cube actually MOVE toward the goal within the episode (steps 0..149 only --
#      sampling at or past 160 reads the post-time-out reset and has corrupted four measurements)?
#
# Reference points measured on this task:
#   null action, effort x20 : gate open 63/64 at t=0, 13/64 at t=149, median tip force 11.2 N
#   trained policy (old)    : gate open  0/64 at every step, force 0.0000 N, cube never moved
#
# Usage:  ./accept_policy.sh <checkpoint.pt> [family] [name]
#   family: ObjectAnywhereEEGrasped (C3, the real carry task) | ObjectPartiallyAssembledEEGrasped
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
REPO=/home/mattephi/Work/Rubitek/UWLab-cubestack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv-is6/bin/python

CK=${1:?usage: accept_policy.sh <checkpoint.pt> [family] [name]}
FAMILY=${2:-ObjectAnywhereEEGrasped}
NAME=${3:-accept}
[ -f "$CK" ] || { echo "no such checkpoint: $CK"; exit 1; }

cd "$REPO" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="/home/mattephi/Work/Rubitek/uwlab-env/compat:$REPO/source/uwlab_assets:$REPO/source/uwlab_tasks:$REPO/source/uwlab:$REPO/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=1073741824
export UWLAB_CUBE_DATASET_DIR=./Datasets_ur10e_delto/OmniReset_pinch2f
export UWLAB_OSC_GRAVITY_COMP=1 UWLAB_OSC_GRAVITY_SIGN=-0.15
export UWLAB_LATCHED_HAND=1 UWLAB_TUCK_OTHERS=0 UWLAB_GRIP_BIAS_RAD=0.05 UWLAB_HAND_EFFORT_SCALE=20
export TRACE_ENVS=64 TRACE_STEPS=160

echo "=== recording $FAMILY from $(basename "$CK")  $(date -Is)"
TRACE_NAME="$NAME" TRACE_CK="$CK" TRACE_FAMILY="$FAMILY" \
  timeout 1800 "$PY" "$CS/trace_full_poses.py" > "$CS/logs/accept_$NAME.log" 2>&1
RC=$?
grep -aE "POSES wrote|Traceback|Error" "$CS/logs/accept_$NAME.log" | tail -3
[ "$RC" -eq 0 ] || { echo "RECORDING FAILED (exit $RC) -- see logs/accept_$NAME.log"; exit 1; }

"$PY" - "$NAME" <<'EOF'
import sys, numpy as np
name = sys.argv[1]
d = np.load(f"/home/mattephi/Work/Rubitek/cube-stack/results/poses_{name}.npz", allow_pickle=True)
c, cp, gp, su = d["contact"], d["cube_pos"], d["goal_pos"], d["success"]
names = [str(x) for x in d["body_names"]]
E = cp.shape[1]
THRESH = 0.2
# Column order follows DELTO_ALL_TIP_NAMES: (1_tip, 5_tip) + (2_tip, 3_tip, 4_tip).
# Getting this wrong once already produced a false "gate never opens even under null action".
THUMB, INDEX = 0, 2

print(f"\n=== ACCEPTANCE: {name}  ({E} envs, steps 0..149 only)")
if c.size == 0:
    print("  CONTACT SENSOR UNAVAILABLE -- cannot judge a grasp. Silence is not evidence of no contact.")
    sys.exit(2)

tot = c.sum(2)
gate = (c[:, :, THUMB] > THRESH) & (c[:, :, INDEX] > THRESH)
disp = np.linalg.norm(cp[:150] - cp[0][None], axis=2)
dist = np.linalg.norm(cp - gp, axis=2)

print("  1. CONTACT FORCE")
for t in (0, 5, 20, 60, 149):
    print(f"     t={t:>3}: median {np.median(tot[t]):7.3f} N   envs with force>0.01 N: {int((tot[t]>0.01).sum()):>2}/{E}")
print(f"     envs with ANY contact at ANY step: {int((tot[:150]>0.01).any(0).sum())}/{E}   peak {tot[:150].max():.2f} N")

print("  2. TWO-FINGER GATE (thumb AND index both > 0.2 N)")
for t in (0, 5, 20, 60, 149):
    print(f"     t={t:>3}: open {int(gate[t].sum()):>2}/{E}")
print(f"     ever open: {int(gate[:150].any(0).sum())}/{E}   median steps open: {int(np.median(gate[:150].sum(0)))}/150")

print("  3. DID THE CUBE MOVE, AND TOWARD THE GOAL?")
print(f"     max displacement within episode: median {np.median(disp.max(0))*1000:6.1f} mm   max {disp.max()*1000:.1f} mm")
print(f"     envs moving cube >5 mm: {int((disp.max(0)>0.005).sum())}/{E}")
print(f"     cube-to-goal: start median {np.median(dist[0])*1000:6.1f} mm -> end {np.median(dist[149])*1000:6.1f} mm")

print("  4. SUCCESS (reported LAST, and only with the above in view)")
print(f"     step 0: {int(su[0].sum())}/{E}     step 149: {int(su[149].sum())}/{E}     delta {int(su[149].sum())-int(su[0].sum()):+d}")

held = int(gate[:150].any(0).sum())
moved = int((disp.max(0) > 0.005).sum())
gained = int(su[149].sum()) - int(su[0].sum())
print("\n  VERDICT:")
print(f"    grasps at all        : {'YES' if held else 'NO '}  ({held}/{E} envs ever close the gate)")
print(f"    moves the cube       : {'YES' if moved else 'NO '}  ({moved}/{E} envs move it >5 mm)")
print(f"    improves on the reset: {'YES' if gained > 0 else 'NO '}  ({gained:+d} successes vs step 0)")
if not held:
    print("    => STILL NOT GRASPING. Do not report a success rate as task performance.")
EOF
