#!/usr/bin/env bash
# SCALE THE DELTO PINCH RESET BANKS on the rented box: record ~300 states per grasped family, replay
# every one of them, and keep only the states still held by thumb+index.
#
# QA IS THE CONTACT VALIDATOR, NOT THE VIEWPORT RENDERER. This container has NVIDIA's GLX/EGL libs
# but was not started with the graphics capability, so Vulkan resolves to llvmpipe and every RTX
# render dies with "vkCreateInstance failed. Vulkan 1.1 is not supported". validate_grasped_bank.py
# needs no rendering: it replays every state in parallel with the arm held by PD (never by per-step
# state writes, which teleport the articulation and drop contact friction) and keeps a state only
# when both fingertips press the cube above the force threshold -- the same criterion the film
# renderer applies, and it filters a whole bank in one run instead of one render per state.
#
# Recording settings are the ones the films validated: cube pinned for the first steps while the grip
# loads, fingertip friction and cube mass as the grasps were sampled, 0.99-quantile acceptance box.
set -uo pipefail
WS=/workspace
cd "$WS/UWLab" || exit 1
PY="$WS/venv/bin/python"
# Refuse to burn rented time on a half-provisioned box: job_fix_isaacsim.sh leaves this marker.
[ -f "$WS/ISAACSIM_OK" ] || { echo "SKIP: /workspace/ISAACSIM_OK missing -- Isaac Sim provisioning not verified"; exit 1; }
DS=Datasets_ur10e_delto/OmniReset_pinch2f
PAIR=InsertiveCube34__ReceptiveCube34
PFX=OmniReset-UR10eDelto-CubeStack
N=${N:-300}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
# $WS/compat first: it carries the uwlab_is6_bridge and the pytorch3d shim that uwlab_tasks imports
# (locally these come from a uwlab_compat.pth pointing at uwlab-env/compat).
export PYTHONPATH="$WS/compat:$WS/UWLab:$WS/UWLab/source/uwlab_tasks:$WS/UWLab/source/uwlab_assets:$WS/UWLab/source/uwlab:$WS/UWLab/source/uwlab_rl"
export UWLAB_DELTO_PINCH=1 UWLAB_PINCH_SEG1_STIFFNESS=10 UWLAB_GPU_COLLISION_STACK_SIZE=2147483648
export UWLAB_CUBE_DATASET_DIR="./$DS"
export UWLAB_DROP_ARM_ACTION=1 UWLAB_ARM_PD_HOLD=1 UWLAB_HOLD_OBJECT_STEPS=8
say(){ echo "SCALE $* $(date -Is)"; }

for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  B=$DS/Resets/$PAIR/resets_$T.pt
  [ -f "$B" ] && [ ! -f "${B%.pt}.local40.pt" ] && cp -f "$B" "${B%.pt}.local40.pt"
  say "RECORD_${T}_START target=$N"
  timeout 5400 "$PY" scripts_v2/tools/record_reset_states.py --task "$PFX-$T-v0" --reset_type "$T" \
      --num_envs 256 --num_reset_states "$N" --headless --dataset_dir "./$DS" > "$WS/logs/rec_$T.log" 2>&1
  rc=$?
  say "RECORD_${T}_END rc=$rc $(grep -ao "[0-9]*/$N \[[0-9:]*" "$WS/logs/rec_$T.log" | tail -1)"
  [ -f "$B" ] || { say "NO_BANK $T"; continue; }
  cp -f "$B" "${B%.pt}.unfiltered.pt"
  say "QA_${T}_START (contact validator, all states in parallel)"
  timeout 3600 "$PY" scripts_v2/tools/validate_grasped_bank.py \
      --reset-states "${B%.pt}.unfiltered.pt" --task "$PFX-$T-v0" --dataset-dir "./$DS" \
      --margin 0.012 --settle-steps 60 --out "$B" > "$WS/logs/qa_$T.log" 2>&1
  say "QA_${T}_END rc=$? $(grep -a 'KEPT' "$WS/logs/qa_$T.log" | tail -1)"
done

say "SCALE_DONE"
for T in ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  "$PY" - <<PYEOF
import torch
p = "$DS/Resets/$PAIR/resets_$T.pt"
try:
    d = torch.load(p, map_location="cpu", weights_only=False)["initial_state"]
    print("$T", len(d["rigid_object"]["insertive_object"]["root_pose"]), "states kept")
except Exception as exc:
    print("$T unreadable:", exc)
PYEOF
done
