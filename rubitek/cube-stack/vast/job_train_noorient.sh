#!/usr/bin/env bash
# Position-only cube stacking. Runs on the rented box under the container's init, via the job loop
# in onstart.sh -- never from an ssh session, which this box family reaps on disconnect.
set -uo pipefail
WS=/workspace
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
cd "$WS/UWLab" || exit 1

# 2000 iterations, not the config's 40000 ceiling. The paper's own learning curves run to
# 2000-2500 updates (Fig. 8 to 2000, Fig. 9 to 2500), so without an explicit cap the run simply
# never stops.
ITERS=${ITERS:-2000}
# 4096 x 32 steps = 131k transitions per update. The paper's §4.3 ablation sweeps 2048..65536 and
# finds small counts still solve near-goal but fail the full task from reaching states, so this is
# the low end of the regime that works. Raise if the card has headroom.
ENVS=${ENVS:-4096}
SEED=${SEED:-42}

"$WS/venv/bin/python" scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-v0 \
  --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --headless \
  env.scene.insertive_object=cube34 \
  env.scene.receptive_object=cube34 \
  env.events.reset_from_reset_states.params.dataset_dir=./Datasets_ur10e_delto/OmniReset

echo "=== train wrapper exited $? (not trusted) ==="
# Positive evidence: a checkpoint must exist and be non-trivial.
find logs/rsl_rl -name 'model_*.pt' -newermt '-1 day' -printf '%s %p\n' | sort -rn | head -5
