#!/usr/bin/env bash
# ONE STAGE of the v2 repose ramp (V2_REPOSE_RECIPE.md sec 2; beads dr-tlx.3, dr-tlx.9, dr-tlx.10).
#
# STAGED, NOT LAUNCHED. The H100s are not free, and R0 (r0_control.sh) must produce P and the O6
# baseline before any stage of this ramp means anything.
#
# The ramp is FIVE SEQUENTIAL JOBS, not an in-run schedule: the mixture fractions are read ONCE, at
# manager-construction time (episode_mixture.py MixtureResetObject.__init__ /
# MixtureGoalPoseCommand.__init__), so an in-run ramp would mean new mutating code inside the module
# whose validators are this campaign's collapse guard. Four extra 92-second startups against a ~40 h
# campaign is 0.25%. This script runs stage N; the advance decision between stages is a human/agent
# reading the certification and the gate proxy, per sec 2.2.
#
#   STAGE=R1 [GPU=0,1] [ITERS=200] [CKPT=...] ramp_stage.sh
#
set -uo pipefail

# ---------------------------------------------------------------------------------------------
# TMP ROOTS -- see r0_control.sh's comment. Without these the job dies before iteration 0 with a
# PermissionError that reads like a job that never launched.
# ---------------------------------------------------------------------------------------------
export UWLAB_TMP_ROOT=${UWLAB_TMP_ROOT:-$HOME/tmp_uwlab}
export TMPDIR=${TMPDIR:-$HOME/tmp_local}
mkdir -p "$UWLAB_TMP_ROOT" "$TMPDIR" || { echo "REFUSING: cannot create tmp roots"; exit 1; }

# ---------------------------------------------------------------------------------------------
# THE RAMP TABLE (V2_REPOSE_RECIPE.md sec 2.1). Four fractions per stage; they must sum to 1.0 and
# classic must stay > 0 or validate_episode_mixture_fractions refuses -- but note that assert is a
# SANITY CHECK, NOT A COLLAPSE GUARD: ep_4300 ran with classic at 0.50 and still fell from 69.5% to
# 12.5% pass@30mm (UWLAB_STATE.md sec 6). That is what A2 exists for.
#
#   stage  transport  classic  low   partial
# ---------------------------------------------------------------------------------------------
case "${STAGE:?set STAGE to R0|R1|R2|R3|R4 or a halved step like R2h}" in
  R0)  TRANSPORT=0.00; CLASSIC=0.50; LOW=0.25; PARTIAL=0.25 ;;
  R1)  TRANSPORT=0.10; CLASSIC=0.45; LOW=0.20; PARTIAL=0.25 ;;
  R2)  TRANSPORT=0.20; CLASSIC=0.40; LOW=0.15; PARTIAL=0.25 ;;
  R3)  TRANSPORT=0.30; CLASSIC=0.35; LOW=0.10; PARTIAL=0.25 ;;
  R4)  TRANSPORT=0.40; CLASSIC=0.25; LOW=0.10; PARTIAL=0.25 ;;
  # Halved steps, for A1/A2's "roll back and halve" path (sec 3.1 step 3).
  R2h) TRANSPORT=0.15; CLASSIC=0.425; LOW=0.175; PARTIAL=0.25 ;;
  R3h) TRANSPORT=0.25; CLASSIC=0.375; LOW=0.125; PARTIAL=0.25 ;;
  R4h) TRANSPORT=0.35; CLASSIC=0.30; LOW=0.10; PARTIAL=0.25 ;;
  *) echo "REFUSING: unknown STAGE='$STAGE'"; exit 1 ;;
esac

# Sum check HERE as well as in python, because a typo in the table above would otherwise surface 92
# seconds later, on the GPU host, as an assertion from a module three layers down.
SUM=$(awk -v a="$TRANSPORT" -v b="$CLASSIC" -v c="$LOW" -v d="$PARTIAL" 'BEGIN{printf "%.6f", a+b+c+d}')
[ "$SUM" = "1.000000" ] || { echo "REFUSING: stage $STAGE fractions sum to $SUM, not 1.0"; exit 1; }

export CUDA_VISIBLE_DEVICES=${GPU:-0,1}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

# Production staging, identical to R0's -- the ramp must differ from its control in the mixture
# fractions and NOTHING ELSE, or the curve is unattributable.
export DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=1
export DEXLIFT_POSE_TILT=0.3
export DEXLIFT_DROP_Z=0.05
export DEXLIFT_EPISODE_MIXTURE=1
export DEXRESET_GATE_PROXY=1
unset DEXLIFT_GOAL_VERTICAL_PROB DEXRESET_C3_RUNG DEXRESET_ST_SPAWN_TIPDOWN
unset DEXLIFT_C4_SEATING_REWARD DEXLIFT_C4_GROSS_UNSEATING_TERM DEXLIFT_C4_AXIAL_REWARD

REPO_DIR=${REPO_DIR:-$HOME/UWLab_ur5edelto}
PYTHON_BIN=${PYTHON_BIN:-$HOME/UWLab/env_uwlab/bin/python}
OUT_DIR=${OUT_DIR:-$HOME/dexreset_v2/ramp_$STAGE}
TASK=${TASK:-DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0}
# NO APOSTROPHE IN THIS MESSAGE, and this is not style. A `'` inside ${VAR:?message} opens a
# single-quoted string that bash swallows up to the NEXT apostrophe anywhere in the file. When the
# count happens to balance, `bash -n` PASSES on a file whose middle has been eaten -- which is what
# happened here: the check() definition and every REFUSING guard below silently became string
# content, and a run with a nonexistent REPO_DIR sailed past all of them. Caught only because a
# later edit flipped the parity into a real syntax error. See test_launcher_scripts.sh, which
# exercises the guards instead of trusting `bash -n`.
CKPT=${CKPT:?set CKPT to the previous stage checkpoint -- R1 starts from ep_3600}
NUM_ENVS=${NUM_ENVS:-32768}
ITERS=${ITERS:-200}

cd "$REPO_DIR" || { echo "REFUSING: REPO_DIR=$REPO_DIR does not exist"; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "REFUSING: PYTHON_BIN=$PYTHON_BIN is not an executable"; exit 1; }
[ -f "$CKPT" ] || { echo "REFUSING: checkpoint $CKPT does not exist"; exit 1; }
case "$(readlink -f "$CKPT")" in
  "$HOME"/ckpt/*) echo "[ramp] warm start from the PRESERVED baseline (read-only): $CKPT" ;;
esac
mkdir -p "$OUT_DIR" || { echo "REFUSING: cannot create OUT_DIR=$OUT_DIR"; exit 1; }
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# ~/ckpt HOLDS THE ONLY COPY OF THE TWO CERTIFIED BASELINES (RESET_SPEC_V2.md sec 6a). Every stage
# writes into its own OUT_DIR; nothing in this campaign may write into ~/ckpt.
case "$(readlink -f "$OUT_DIR")/" in
  "$HOME"/ckpt/*) echo "REFUSING: OUT_DIR is inside ~/ckpt, which holds the only copy of the certified baselines."; exit 1 ;;
esac

# ---------------------------------------------------------------------------------------------
# THE UPDATE BUDGET (bead dr-tlx.9, V2_REPOSE_RECIPE.md sec 2.3). THIS IS THE LINE THAT MATTERS.
# ---------------------------------------------------------------------------------------------
# The measured collapse -- "55 percent of the skill gone in 50 epochs" (goal_mixture.py:27) -- was
# at num_envs 4096 / minibatch 36864 / horizon 36, i.e. 4 minibatches x 5 mini_epochs = 20 optimiser
# updates per iteration. 50 epochs is therefore 1000 UPDATES, not 50 of anything transferable.
#
# At 32768 envs with the shipped minibatch 18432 one iteration is 32768*36/18432 = 64 minibatches
# x 5 = 320 updates. The whole measured collapse budget would be 3.1 iterations -- before the first
# checkpoint at the shipped save_frequency 50. Iterations is the wrong unit.
#
# minibatch 73728 gives 32768*36/73728 = 16 minibatches x 5 = 80 updates/iteration, putting the
# budget at 12.5 iterations, and save_frequency 5 puts a rollback point inside it.
MINIBATCH=${MINIBATCH:-73728}
SAVE_FREQ=${SAVE_FREQ:-5}
HORIZON=36
PRODUCT=$(( NUM_ENVS * HORIZON ))
[ $(( PRODUCT % MINIBATCH )) -eq 0 ] || {
  echo "REFUSING: minibatch $MINIBATCH does not divide num_envs*horizon = $PRODUCT."
  echo "  rl_games requires an integer number of minibatches; a non-divisor silently reshapes the batch."
  exit 1
}
echo "[ramp] update budget: $(( PRODUCT / MINIBATCH )) minibatches x 5 mini_epochs = $(( PRODUCT / MINIBATCH * 5 )) updates/iteration"
echo "[ramp] measured collapse budget is ~1000 updates => ~$(( 1000 / (PRODUCT / MINIBATCH * 5) )) iterations. save_frequency=$SAVE_FREQ."

# ---------------------------------------------------------------------------------------------
# wandb: OFFLINE + periodic sync (RESET_SPEC_V2.md sec 5 -- the H100 link is flaky). NO WEIGHTS.
# ---------------------------------------------------------------------------------------------
export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR:-$OUT_DIR/wandb}
# One GROUP for all five stage runs, so the ramp reads as one campaign rather than five unrelated
# runs. rl_games' train.py calls wandb.init() with no `group=` argument, and wandb honours this
# environment variable natively -- so this needs no code change to train.py.
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-dexreset_v2_repose_ramp_$(date +%Y%m%d)}
mkdir -p "$WANDB_DIR"
# NO WEIGHT LOGGING (RESET_SPEC_V2.md sec 4): train.py's wandb.init passes save_code=True, which
# uploads SOURCE, not weights, and nothing here calls wandb.watch or logs a .pth artifact. Stated
# so a reader does not have to re-derive it from train.py.

LOG="$OUT_DIR/train_$STAGE.log"
echo "[ramp] stage=$STAGE transport=$TRANSPORT classic=$CLASSIC low=$LOW partial=$PARTIAL iters=$ITERS"
echo "[ramp] GPU=$CUDA_VISIBLE_DEVICES -- confirm BOTH are free with nvidia-smi before running"

# ---------------------------------------------------------------------------------------------
# 2-GPU LAUNCH -- UNVERIFIED FOR rl_games ON THIS BOX. READ THIS BEFORE SETTING NPROC=2.
# ---------------------------------------------------------------------------------------------
# V2_POSE_FINDINGS.md F19 established a WORKING 2-GPU configuration ("repaired venv, no preload",
# test5, 2 logged PPO iterations) -- but it did so for `scripts/reinforcement_learning/rsl_rl/
# train.py`. This campaign trains with rl_games, because ep_3600 is an rl_games checkpoint and the
# certified lineage is rl_games. rl_games' train.py HAS a --distributed flag, but the NCCL path
# under it has not been exercised on this box, and F19's failure mode was an
# `ncclUnhandledCudaError` raised from inside the trainer's own parameter broadcast -- i.e. after
# startup, in trainer-specific code, which is exactly the part that differs between the two.
#
# DEFAULT IS 1 GPU. Confirm two ranks log iterations before committing a multi-day campaign to it.
#
# AND THE 1-GPU CASE IS NOT THE SAME RUN, SLOWER (V2_REPOSE_RECIPE.md sec 6.2). Measured
# 48-54 GB/rank at 16384 envs per rank (UWLAB_STATE.md:114), so ONE 80 GB H100 holds 16384 envs,
# not 32768 -- launching the 2-GPU env count on one card OOMs partway through scene build. Halving
# the envs also halves the optimiser updates per iteration (16384*36/73728 = 8 minibatches x 5 = 40,
# against 80), so the stage caps have to be translated by UPDATE BUDGET, not by wall clock: a
# 200-iteration cap at 32768 envs is 16000 updates, which is 400 iterations at 16384. That is
# sec 2.3's own point -- iterations is the wrong unit -- applied to the fallback.
NPROC=${NPROC:-1}
SINGLE_GPU_MAX_ENVS=16384
if [ "$NPROC" -le 1 ] && [ "$NUM_ENVS" -gt "$SINGLE_GPU_MAX_ENVS" ]; then
  echo "REFUSING: NPROC=$NPROC with NUM_ENVS=$NUM_ENVS."
  echo "  Measured 48-54 GB/rank at 16384 envs/rank; $NUM_ENVS envs on one 80 GB card does not fit."
  echo "  Set NUM_ENVS=$SINGLE_GPU_MAX_ENVS -- and RE-DERIVE the stage cap rather than reusing it:"
  echo "    at 16384 envs, $(( SINGLE_GPU_MAX_ENVS * HORIZON / MINIBATCH * 5 )) updates/iteration,"
  echo "    so this stage's ITERS should be $(( ITERS * 2 )) to hold the same update budget"
  echo "    ($(( ITERS * NUM_ENVS * HORIZON / MINIBATCH * 5 )) updates). See V2_REPOSE_RECIPE.md sec 6.2."
  echo "  Refusing rather than silently halving either number: which one to change is a plan"
  echo "  decision, and guessing it would make the run incomparable to the other stages."
  exit 1
fi
if [ "$NPROC" -gt 1 ]; then
  echo "[ramp] WARNING: NPROC=$NPROC. rl_games multi-GPU is UNVERIFIED on this box (F19 verified"
  echo "[ramp]          rsl_rl, not rl_games). Confirm two ranks log iterations in the smoke first."
  LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC")
  DIST_FLAG=(--distributed)
else
  LAUNCH=("$PYTHON_BIN" -u)
  DIST_FLAG=()
fi

"${LAUNCH[@]}" scripts/reinforcement_learning/rl_games/train.py \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" \
    --seed "${SEED:-1}" \
    --headless \
    "${DIST_FLAG[@]}" \
    --track True \
    --wandb-project-name "${WANDB_PROJECT:-dexreset-v2}" \
    --wandb-entity "${WANDB_ENTITY:-i_domrachev-interactive-robotic-systems-lab-kaist}" \
    --wandb-name "dexreset_v2_repose_${STAGE}_p${TRANSPORT}" \
    env.transport_goal_prob="$TRANSPORT" \
    env.classic_goal_prob="$CLASSIC" \
    env.low_goal_prob="$LOW" \
    env.partial_assembly_prob="$PARTIAL" \
    env.transport_goal_tilt=0.45 \
    env.transport_goal_z="[0.106203,0.306203]" \
    agent.params.config.minibatch_size="$MINIBATCH" \
    agent.params.config.save_frequency="$SAVE_FREQ" \
    > "$LOG" 2>&1
RC=$?
echo "[ramp] train rc=$RC"

# ---------------------------------------------------------------------------------------------
# BANNER READBACK, fail-closed. RESET_SPEC_V2.md sec 1a trap 3 and R5: read the staged value out of
# the run log, never trust the command line. dr-tlx.9 explicitly requires minibatch and
# save_frequency to be read back BEFORE the ramp is believed.
# ---------------------------------------------------------------------------------------------
FAILED=0
check() {
  local label="$1" pat="$2" got
  got=$(grep -aoE "$pat" "$LOG" | head -1)
  if [ -n "$got" ]; then
    echo "[ramp] $label: $got"
  else
    echo "BANNER MISSING [$label]: /$pat/"
    FAILED=1
  fi
}

check "pose tilt"   "\[dexlift\] POSE_TILT staged: .*\+-0\.3000 rad.*\[0, 0\.050\] m"
check "gravity"     "\[dexlift\] gravity PINNED at .*-9\.81"
check "fractions"   "classic_goal_prob=[0-9.]+ .*transport_goal_prob=[0-9.]+"
check "gate proxy"  "\[dexreset\] GATE PROXY staged:.*"
if [ "$STAGE" != R0 ]; then
  check "transport"  "\[dexlift\] TRANSPORT GOAL branch staged: $TRANSPORT.*"
fi
grep -aqE "\[dexlift\] GOAL VERTICAL MIXTURE staged" "$LOG" && {
  echo "BANNER UNEXPECTED [goal-vertical]: DEXLIFT_GOAL_VERTICAL_PROB took effect. F38 says it"
  echo "  cannot compose with the episode mixture; this run is not the configuration intended."
  FAILED=1; }

# dr-tlx.9's readback, from the run's OWN saved config rather than from this script's variables.
AGENT_YAML=$(find "$OUT_DIR" -name agent.yaml -path '*params*' 2>/dev/null | head -1)
if [ -n "$AGENT_YAML" ]; then
  echo "[ramp] saved agent cfg: $AGENT_YAML"
  grep -aE "minibatch_size|save_frequency|horizon_length|mini_epochs" "$AGENT_YAML" | sed 's/^/[ramp]   /'
  grep -aq "minibatch_size: $MINIBATCH" "$AGENT_YAML" || { echo "READBACK FAILED: minibatch_size is not $MINIBATCH in $AGENT_YAML"; FAILED=1; }
  grep -aq "save_frequency: $SAVE_FREQ" "$AGENT_YAML" || { echo "READBACK FAILED: save_frequency is not $SAVE_FREQ in $AGENT_YAML"; FAILED=1; }
else
  echo "READBACK FAILED: no params/agent.yaml found under $OUT_DIR -- cannot confirm the update budget (dr-tlx.9)."
  FAILED=1
fi

grep -aE "Traceback|Error|ncclUnhandled|inference tensor" "$LOG" | tail -12
[ "$FAILED" = 0 ] || { echo "[ramp] READBACK/BANNER ASSERTIONS FAILED -- do not advance the ramp on this run."; exit 1; }
[ "$RC" = 0 ] || { echo "[ramp] trainer exited $RC."; exit 1; }

echo "[ramp] stage $STAGE done. BEFORE ADVANCING (sec 2.2):"
echo "[ramp]   (a) certify: run r0_control.sh cert against this stage's last checkpoint; need pass@30mm >= 0.85*P"
echo "[ramp]   (b) plateau: GateProxy/passive_three_atend_frac/transport risen < 0.02 over the last 100 iters"
echo "[ramp]   A2 abort if pass@30mm < 0.70*P. A1 abort on classic-kind success -5pts/25iters while reward flat or rising."
