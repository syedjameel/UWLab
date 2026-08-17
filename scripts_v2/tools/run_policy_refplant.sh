#!/usr/bin/env zsh
# Vendored into this repo 2026-08-17 (bead UWLab-qiao.1 follow-on). Copied verbatim from
# DL_A6000:/home/dom_iva/run_policy_refplant.sh, UNCHANGED by this task -- it already carried the
# four DEXLIFT_REF_* exports (this is where the fix applied to the other three launchers was copied
# from) and does NOT carry the DEXLIFT_SPAWN_CLEARANCE line; see the accompanying report for why it
# was left that way rather than added. Until this copy, that DL_A6000 home directory was the ONLY
# place this script existed: no git, no version history. Vendored now because training is moving to
# a new host (a parked vast.ai 1x5090) that has never seen this file. Every absolute path and host
# assumption baked into this copy is still DL_A6000-specific -- see the accompanying report for the
# full list; nothing below has been parameterized.
set -x
cd /home/dom_iva/UWLab_ur5edelto
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0
LOG="/home/dom_iva/UWLab_ur5edelto/policy_refplant_bounded.log"
CKPT="/home/dom_iva/UWLab_ur5edelto/logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_lift/2026-08-16_08-24-59/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_lift_ep_1950_rew_22.796772.pth"
timeout -s KILL 1200 /home/dom_iva/UWLab/env_uwlab/bin/python -u scripts_v2/tools/generate_reset_states_policy.py \
  --checkpoint "$CKPT" \
  --num_envs 128 --smoke_steps 2000 --headless \
  > "$LOG" 2>&1
echo "EXIT_CODE=$?" >> "$LOG"
