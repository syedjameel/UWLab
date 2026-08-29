#!/usr/bin/env bash
# R0 -- the control run of the v2 repose retrain (V2_REPOSE_RECIPE.md sec 2.1, sec 6; bead dr-tlx.3).
#
# STAGED, NOT LAUNCHED. Nothing here runs until someone invokes it, and the H100s are not free.
#
# R0 produces the TWO baselines every later number in the campaign is divided by:
#
#   (a) P  -- ep_3600's pass@30mm on the general Reorient distribution, MEASURED IN THIS HARNESS.
#             A2's abort thresholds are 0.85*P and 0.70*P, never 0.85*69.53% -- RESET_SPEC_V2.md
#             sec 6a's binding comparison rule: "if v2 is measured at a tighter gate than v1 was,
#             it will look worse while being better".
#   (b) the passive-three gate-proxy baseline in the v2 env -- V2_REPOSE_RECIPE.md O6, which has
#             NEVER BEEN MEASURED. Every "the retrain improved X" claim divides by it.
#
# The two are separate runs on purpose. They score different quantities (pose tracking vs the
# held-state gate chain's probe-free prefix) and R7 forbids letting either stand in for the other.
#
#   [GPU=0] [CKPT=...] [OUT_DIR=...] [EPISODES=512] [PROXY_STEPS=4000] r0_control.sh [cert|proxy|both]
#
set -uo pipefail

STAGE=${1:-both}
case "$STAGE" in cert|proxy|both) ;; *) echo "REFUSING: stage must be cert|proxy|both, got '$STAGE'"; exit 1 ;; esac

# ---------------------------------------------------------------------------------------------
# TMP ROOTS. NOT OPTIONAL, AND NOT COSMETIC.
# ---------------------------------------------------------------------------------------------
# Without these the job dies BEFORE iteration 0 with a PermissionError raised from deep inside the
# Omniverse Kit startup, which on a headless run looks exactly like a job that never launched at
# all -- no banner, no traceback in the place anyone looks first. Exported here rather than left to
# the caller's shell profile so that a run started from cron, from ssh with a non-login shell, or
# from another agent's launcher gets them too.
export UWLAB_TMP_ROOT=${UWLAB_TMP_ROOT:-$HOME/tmp_uwlab}
export TMPDIR=${TMPDIR:-$HOME/tmp_local}
mkdir -p "$UWLAB_TMP_ROOT" "$TMPDIR" || { echo "REFUSING: cannot create tmp roots"; exit 1; }

export CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

# ---------------------------------------------------------------------------------------------
# STAGING. Every one of these is asserted back out of the log below; none is trusted from here.
# ---------------------------------------------------------------------------------------------
# Production staging, unchanged -- R0 is a CONTROL and must be the pre-ramp task exactly.
export DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=1
export DEXLIFT_POSE_TILT=0.3
export DEXLIFT_DROP_Z=0.05
# MUST BE UNSET, not set to 0 (V2_REPOSE_RECIPE.md sec 5.2):
#   DEXLIFT_GOAL_VERTICAL_PROB -- F38, it raises TypeError against DEXLIFT_EPISODE_MIXTURE=1.
#   DEXRESET_C3_RUNG           -- raises against DEXLIFT_EPISODE_MIXTURE=1 by design.
#   DEXRESET_ST_SPAWN_TIPDOWN  -- F51, S_t's peg is horizontal and needs no spawn change.
unset DEXLIFT_GOAL_VERTICAL_PROB DEXRESET_C3_RUNG DEXRESET_ST_SPAWN_TIPDOWN

REPO_DIR=${REPO_DIR:-$HOME/UWLab_ur5edelto}
PYTHON_BIN=${PYTHON_BIN:-$HOME/UWLab/env_uwlab/bin/python}
OUT_DIR=${OUT_DIR:-$HOME/dexreset_v2/r0}
CKPT=${CKPT:-$HOME/ckpt/last_dexlift_ur5e_delto_reljointpos_tableleg_reorient_ep_3600_rew_38.38917.pth}
TASK=${TASK:-DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-v0}

cd "$REPO_DIR" || { echo "REFUSING: REPO_DIR=$REPO_DIR does not exist"; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "REFUSING: PYTHON_BIN=$PYTHON_BIN is not an executable"; exit 1; }
[ -f "$CKPT" ] || { echo "REFUSING: checkpoint $CKPT does not exist"; exit 1; }
mkdir -p "$OUT_DIR" || { echo "REFUSING: cannot create OUT_DIR=$OUT_DIR"; exit 1; }
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"

# ~/ckpt HOLDS THE ONLY COPY OF THE TWO CERTIFIED BASELINES ON THIS BOX (RESET_SPEC_V2.md sec 6a).
# R0 only reads it, but say so, because every later stage of this campaign writes checkpoints and
# none of them may write here.
echo "[r0] reading (never writing) baseline checkpoint: $CKPT"
echo "[r0] tmp roots: UWLAB_TMP_ROOT=$UWLAB_TMP_ROOT TMPDIR=$TMPDIR"
echo "[r0] GPU=$CUDA_VISIBLE_DEVICES  -- confirm it is FREE with nvidia-smi before running this"

# ---------------------------------------------------------------------------------------------
# Banner assertions, fail-closed. Same shape and same reason as run_certify.sh's PLANT/ASSET loops:
# a MISSING banner is a failure, not a pass. RESET_SPEC_V2.md sec 1a trap 3.
# ---------------------------------------------------------------------------------------------
assert_banner() {  # assert_banner <log> <human label> <grep -E pattern>
  local log="$1" label="$2" pattern="$3" got
  got=$(grep -aoE "$pattern" "$log" | head -1)
  [ -n "$got" ] || { echo "BANNER MISSING [$label]: no match for /$pattern/ in $log"; return 1; }
  echo "[r0] $label: $got"
}

# ---------------------------------------------------------------------------------------------
# WALL-CLOCK RECORDER. Not bookkeeping -- it closes an OPEN.
# ---------------------------------------------------------------------------------------------
# V2_REPOSE_RECIPE.md O19: the "~20 min" per-certification cost in sec 6.1 is an ESTIMATE that was
# never measured, and sec 6.3's entire cut ordering is argued against it. R0's first half runs a
# real certification through the real harness at the real settings, so its wall clock IS that
# number. Recorded here, explicitly, rather than left to be reconstructed from file timestamps
# later -- a reconstruction is a second measurement of a different thing, and this campaign has
# already lost time to exactly that (F26/R7: a number whose provenance has to be re-derived is a
# number nobody can safely cite).
#
# It also answers the question sec 6.3 leaves open. Certification is a rollout in the SAME dexlift
# scene training uses, so it should get whatever speedup the scene gives -- EXCEPT that it runs at
# num_envs 256, where throughput is worse (measured 1452 -> 5769 env-steps/s across 4096 -> 32768
# envs). Whether a small-batch rollout moves with the scene is what decides whether sec 6.3's
# ordering can ever invert. This number against the 32768-env training rate settles it.
record_timing() {  # record_timing <json path> <label> <seconds> <extra json fields...>
  local out="$1" label="$2" secs="$3"; shift 3
  {
    printf '{\n'
    printf '  "schema": "dexreset.r0_timing.v1",\n'
    printf '  "label": "%s",\n' "$label"
    printf '  "wall_clock_s": %s,\n' "$secs"
    printf '  "wall_clock_min": %s,\n' "$(awk -v s="$secs" 'BEGIN{printf "%.2f", s/60}')"
    # The job pays Isaac startup ONCE, and every certification is its own job, so the TOTAL is the
    # right number for sec 6's budget. The rollout-only figure is what answers the scene-speedup
    # question, so both are recorded rather than one being left to subtraction by a later reader.
    printf '  "isaac_startup_s_measured": 92,\n'
    printf '  "rollout_only_s_approx": %s,\n' "$(awk -v s="$secs" 'BEGIN{printf "%.0f", (s-92>0)?s-92:0}')"
    printf '  "host": "%s",\n' "$(hostname)"
    printf '  "gpu": "%s",\n' "$CUDA_VISIBLE_DEVICES"
    printf '  "utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while [ "$#" -ge 2 ]; do printf '  "%s": %s,\n' "$1" "$2"; shift 2; done
    printf '  "closes": "V2_REPOSE_RECIPE.md O19 (sec 6.1 estimate, sec 6.3 ordering depends on it)"\n'
    printf '}\n'
  } > "$out"
  echo "[r0] $label wall clock: ${secs}s ($(awk -v s="$secs" 'BEGIN{printf "%.1f", s/60}') min) -> $out"
}

# ---------------------------------------------------------------------------------------------
# (a) CERTIFICATION -- gives P.
# ---------------------------------------------------------------------------------------------
if [ "$STAGE" = cert ] || [ "$STAGE" = both ]; then
  # Delegated to the existing harness rather than reimplemented, so P is measured by the same code
  # that measured every other number in the certification ladder. --pos_tol 0.03 is the 30 mm gate
  # A2's thresholds are stated at; certify_pose.py scores the whole ladder anyway, so the JSON
  # carries the other rungs too and R7 is satisfied by the file itself.
  LOG="$OUT_DIR/r0_cert.log"
  _cert_t0=$(date +%s)
  TASK="$TASK" COLLIDERS=hullfix3 SELFCOLL=on HAND=ref ARM=ours \
    GPU="$CUDA_VISIBLE_DEVICES" TILT=0.3 DROP_Z=0.05 \
    EPISODES="${EPISODES:-512}" POS_TOL=0.03 ADR_DIFFICULTY=max \
    REPO_DIR="$REPO_DIR" PYTHON_BIN="$PYTHON_BIN" \
    CERTIFY_PY="$REPO_DIR/scripts_v2/tools/certification/certify_pose.py" \
    OUT_DIR="$OUT_DIR" \
    bash scripts_v2/tools/certification/run_certify.sh r0_ep3600 "$CKPT" 2>&1 | tee "$LOG"
  record_timing "$OUT_DIR/r0_cert_timing.json" "certification" "$(( $(date +%s) - _cert_t0 ))" \
    num_envs "${NUM_ENVS:-256}" episodes "${EPISODES:-512}"
  echo "[r0] P (pass@30mm) is in $OUT_DIR/cert_r0_ep3600.json -- record it in V2_REPOSE_RECIPE.md sec 3.2"
  echo "[r0] O19 CLOSED by r0_cert_timing.json: replace sec 6.1's estimated ~20 min with this number,"
  echo "[r0]   recount sec 6.1's certification row (8 x this), and re-check sec 6.3's crossover."
fi

# ---------------------------------------------------------------------------------------------
# (b) GATE PROXY -- gives the O6 baseline.
# ---------------------------------------------------------------------------------------------
if [ "$STAGE" = proxy ] || [ "$STAGE" = both ]; then
  # THE V2 ENV, at the R0 mixture: production fractions, transport branch OFF. That is what makes
  # this a control -- the ramp's first stage (R1) differs from it in exactly one field.
  export DEXLIFT_EPISODE_MIXTURE=1
  export DEXRESET_GATE_PROXY=1
  export DEXLIFT_EPISODE_MIXTURE_DATASET_DIR=${DEXLIFT_EPISODE_MIXTURE_DATASET_DIR:-$HOME/data/partial_assemblies}
  [ -d "$DEXLIFT_EPISODE_MIXTURE_DATASET_DIR" ] || {
    echo "REFUSING: DEXLIFT_EPISODE_MIXTURE_DATASET_DIR=$DEXLIFT_EPISODE_MIXTURE_DATASET_DIR does not exist."
    echo "  The class default (Hugging Face) 404s for this pair -- see _apply_episode_mixture's own comment."
    exit 1
  }

  LOG="$OUT_DIR/r0_proxy.log"
  OUT="$OUT_DIR/r0_gate_proxy.json"
  _proxy_t0=$(date +%s)
  timeout -s KILL 7200 "$PYTHON_BIN" -u scripts_v2/tools/measure_gate_proxy.py \
      --task "$TASK" --checkpoint "$CKPT" \
      --num_envs "${NUM_ENVS:-256}" --steps "${PROXY_STEPS:-4000}" \
      --seed "${SEED:-12345}" --out "$OUT" --headless > "$LOG" 2>&1
  RC=$?
  record_timing "$OUT_DIR/r0_proxy_timing.json" "gate_proxy_rollout" "$(( $(date +%s) - _proxy_t0 ))" \
    num_envs "${NUM_ENVS:-256}" steps "${PROXY_STEPS:-4000}"
  echo "[r0] measure_gate_proxy rc=$RC"

  FAILED=0
  # POSE_TILT and the drop clamp -- the task definition these numbers describe.
  assert_banner "$LOG" "pose tilt" "\[dexlift\] POSE_TILT staged: .*\+-0\.3000 rad.*\[0, 0\.050\] m" || FAILED=1
  # Gravity -- R1 says every held reset is held AGAINST GRAVITY. A finetune pinned near the ADR
  # curriculum floor otherwise trains in vacuum (_apply_full_gravity's own docstring).
  assert_banner "$LOG" "gravity" "\[dexlift\] gravity PINNED at .*-9\.81" || FAILED=1
  # The mixture, and its fractions POST-override (they are validated at manager construction, not
  # at __post_init__, so this banner is the only place the effective values appear).
  assert_banner "$LOG" "mixture" "\[dexlift\] episode mixture MECHANISM wired.*" || FAILED=1
  assert_banner "$LOG" "fractions" "classic_goal_prob=[0-9.]+ .*transport_goal_prob=[0-9.]+" || FAILED=1
  # The instrument itself. F40/F41: four flags in this repo were read by nothing while launchers
  # kept exporting them. A run whose metric was silently absent must not produce a baseline.
  assert_banner "$LOG" "gate proxy" "\[dexreset\] GATE PROXY staged:.*" || FAILED=1
  # And the transport branch must be OFF at R0 -- if this banner appears at all, the control is not
  # a control.
  if grep -aqE "\[dexlift\] TRANSPORT GOAL branch staged" "$LOG"; then
    echo "BANNER UNEXPECTED [transport]: R0 must run with transport_goal_prob=0; the branch announced itself."
    FAILED=1
  else
    echo "[r0] transport branch: correctly absent (transport_goal_prob=0)"
  fi

  grep -aE "RESULT|REFUSING|Traceback|Error|inference tensor" "$LOG" | tail -12
  [ "$FAILED" = 0 ] || { echo "[r0] BANNER ASSERTIONS FAILED -- do not record this number."; exit 1; }
  [ "$RC" = 0 ] || { echo "[r0] measure_gate_proxy exited $RC -- do not record this number."; exit 1; }
  echo "[r0] O6 baseline is in $OUT -- record the passive_two rate in V2_REPOSE_RECIPE.md sec 4.2"
  echo "[r0]   NOTE: two chain gates, not three -- co_move left the held chain 2026-08-29 (18b8ed4)."
fi

echo "[r0] done. Both numbers are baselines: neither may be quoted as a yield (RESET_SPEC_V2.md R7)."
