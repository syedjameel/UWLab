#!/usr/bin/env bash
# STIFF-ARM CURRICULUM. Same 2x2 as before, on an arm that can actually move.
#
# WHY THE PLANT CHANGES. Measured with a scripted controller and no policy anywhere in the loop:
# driving the hand toward a fixed point 196.6 mm away at a full-scale command -- 20 mm per control
# step -- the end effector closes 56 mm in 160 steps. That is 0.35 mm/step, under 2% of what the
# action space advertises. At the package's own eval gains it is 1.4 mm/step, 4x better and still
# 7%. Every family whose cube has to travel is capped by that: a 209 mm carry needs more
# displacement than a 160-step episode can deliver, which is why Stable-Grasp and Reaching have
# scored a flat 0.0000 on every run while Near-Goal -- 13 mm of travel, cube already resting on the
# base -- trains normally to 0.20.
#
# It is not friction and not torque: joint friction and armature read identically zero off the
# constructed scene, effort limits are 330/330/150/56/56/56 N*m, and no sysid term is wired into
# these tasks at all (randomize_arm_from_sysid_fixed lives in the Finetune configs, which nothing
# here inherits). It is the gains. motion_stiffness 200 N/m at a damping ratio of 3.0 -- three times
# critically damped -- on a UR10e carrying a 1.77 kg hand.
#
# On the scripted carry the change is worth roughly 3x on final placement error (194.9 -> 69.4 mm).
# Doubling stiffness again buys 3 mm, and under-damping is worse, so 1000/1.0 is the knee.
#
# NO SEEDING. This is a plant change: a policy trained against the soft arm has learned a different
# transfer function, so its weights are not a head start. Every box restarts at stage A.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

UV="$HOME/.local/bin/uv"
PY="$WS/venv/bin/python"
if ! "$PY" -c "import inspect, rsl_rl.modules.actor_critic as m, sys; sys.exit(0 if 'gsde' in inspect.getsource(m) else 1)"; then
  echo "=== repairing rsl_rl: installing the UW-Lab gSDE fork ==="
  "$UV" pip install --python "$PY" --no-deps --force-reinstall \
    "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@main" || exit 1
  "$UV" pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 1
fi

ENVS=${ENVS:-2048}
SEED=${SEED:-42}
ENTROPY=${ENTROPY:-0.0005}
TASK=${TASK:-OmniReset-UR10eDelto-CubeStackTwoFingerGrip-RelCartesianOSC-State-v0}
RUNS=logs/rsl_rl/ur10e_delto_omnireset_agent

C4=ObjectPartiallyAssembledEEGrasped
C3=ObjectAnywhereEEGrasped
C2=ObjectRestingEEGrasped
C1=ObjectAnywhereEEAnywhere

newest_ck () { find "$RUNS" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }

stage () {  # stage <name> <iters> <types> <probs> <resume-ck>
  local name=$1 iters=$2 types=$3 probs=$4 ck=$5
  echo
  echo "=== STAGE $name : $iters iters, families $types ==="
  local resume=()
  [ -n "$ck" ] && [ -f "$ck" ] && { resume=(--resume_path "$ck"); echo "=== resuming from $ck ==="; }
  "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$iters" --seed "$SEED" --headless \
    "${resume[@]}" --run_name "grip_$name" \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset \
    "env.actions.arm.motion_stiffness=[1000.0,1000.0,1000.0,3.0,3.0,3.0]" \
    "env.actions.arm.motion_damping_ratio=[1.0,1.0,1.0,0.1,0.1,0.1]" \
    "env.events.reset_from_reset_states.params.reset_types=$types" \
    "env.events.reset_from_reset_states.params.probs=$probs" \
    agent.algorithm.entropy_coef="$ENTROPY" \
    agent.save_interval=50 2>&1 | tr '\r' '\n' | grep -vE '^\[[0-9]'
  echo "=== stage $name done, newest: $(newest_ck) ==="
}

# STAGE B IS THE ONE THAT MATTERS NOW. C3 is the family the grip fix unblocks: the cube starts in
# the hand 209 mm from the goal, there is no release or zero-velocity requirement in the success
# predicate, and a policy that simply holds and transports scores. Under the old action term the
# cube was on the table long before it could get there. Budget accordingly -- A is a short warm
# start on the family that already trains, and D is where Reaching finally enters.
# START_CK, when set, is an EXPLICIT path -- never "the newest checkpoint on the box". Defaulting
# it to newest once silently resumed stage B from a previous run's untrained model_0 while printing
# "resuming from...", which is indistinguishable from a real resume in the log. An unset START_CK
# means stage A trains from scratch, which is correct for a fresh box.
CK0=${START_CK:-}
if [ -n "$CK0" ]; then
  echo "=== refusing START_CK: this job changes the OSC gains, so a checkpoint trained against the"
  echo "=== soft arm is not a head start. Unset START_CK to run it."
  exit 1
fi
if [ -n "$CK0" ] && [ ! -f "$CK0" ]; then
  echo "=== START_CK=$CK0 does not exist; refusing to silently train from scratch ==="
  exit 1
fi
if [ -n "$CK0" ]; then
  echo "=== stage A skipped: seeding from $CK0 ==="
  CK=$CK0
else
  stage A "${ITERS_A:-120}" "[$C4]"               "[1.0]"                 ""      ; CK=$(newest_ck)
fi
stage B "${ITERS_B:-400}" "[$C4,$C3]"             "[0.4,0.6]"             "$CK"   ; CK=$(newest_ck)
stage C "${ITERS_C:-300}" "[$C4,$C3,$C2]"         "[0.3,0.4,0.3]"         "$CK"   ; CK=$(newest_ck)
stage D "${ITERS_D:-900}" "[$C4,$C3,$C2,$C1]"     "[0.25,0.25,0.25,0.25]" "$CK"

echo "=== grip curriculum done ==="
find "$RUNS" -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -8
