#!/usr/bin/env zsh
# Vendored into this repo 2026-08-17 (bead UWLab-qiao.1 follow-on). Copied verbatim, post-fix, from
# DL_A6000:/home/dom_iva/run_policy_reset_states.sh -- until this copy, that DL_A6000 home directory
# was the ONLY place this script existed: no git, no version history. Vendored now because training
# is moving to a new host (a parked vast.ai 1x5090) that has never seen this file. Every absolute
# path and host assumption baked into this copy is still DL_A6000-specific -- see the accompanying
# report for the full list; nothing below has been parameterized.
set -x
cd /home/dom_iva/UWLab_ur5edelto
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl"
# REFERENCE-PLANT VARS (bead UWLab-qiao.1 follow-on). Missing here previously turned a 46.71%
# acceptance run into a 2.69% one: the generator silently built the IDENTIFIED hand (effort
# 0.06-0.17 N.m, velocity 3.0 rad/s) against a checkpoint trained under the REFERENCE block (30.0
# N.m, 10000 rad/s), so ~6 rad/s of commanded finger closure hit a 3.0 rad/s ceiling and could not
# track at all. See generate_reset_states_policy.py's five (now six) [verify] lines, which read
# these back off the CONSTRUCTED env_cfg rather than trusting this export block.
export DEXLIFT_REF_RESET=1
export DEXLIFT_REF_ACTUATORS=1
export DEXLIFT_REF_HAND_ACT=1
export DEXLIFT_REF_ARM_ACT=0
# CLEARANCE-AWARE LOW SPAWN toggle (bead UWLab-qiao.1). Default OFF so existing measurements stay
# reproducible; override by exporting DEXLIFT_SPAWN_CLEARANCE=1 in the CALLER's environment before
# invoking this script.
export DEXLIFT_SPAWN_CLEARANCE="${DEXLIFT_SPAWN_CLEARANCE:-0}"
LOG="/home/dom_iva/UWLab_ur5edelto/policy_reset_states_lift.log"
CKPT="/home/dom_iva/UWLab_ur5edelto/logs/rl_games/dexlift_ur5e_delto_reljointpos_tableleg_lift/2026-08-16_08-24-59/nn/last_dexlift_ur5e_delto_reljointpos_tableleg_lift_ep_1950_rew_22.796772.pth"
timeout -s KILL 1800 /home/dom_iva/UWLab/env_uwlab/bin/python -u scripts_v2/tools/generate_reset_states_policy.py \
  --checkpoint "$CKPT" \
  --num_envs 128 --num_reset_states 300 --headless \
  > "$LOG" 2>&1
echo "EXIT_CODE=$?" >> "$LOG"
