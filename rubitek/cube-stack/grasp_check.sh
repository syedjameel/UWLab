#!/usr/bin/env bash
# Do the fingers CLOSE now? A visual check that does not wait for a success rate to move.
#
# WHY THIS IS THE RIGHT CHECK. The full task scored 0.0% because closing the hand was worth exactly
# zero reward: `ee_asset_distance` reads only the palm body, so parking the open hand on the cube was
# already optimal. The recordings show precisely that -- the arm descends, the palm arrives, the
# fingers never close, and the cube does not move for 160 steps.
#
# The fix adds `fingertip_object_distance` (dense, pulls the tips onto the cube) ahead of
# `any_finger_contact` and `grasp_contact`. Whether it works is visible in the frames LONG before it
# is visible in a success rate: a policy that closes its fingers on the cube but cannot yet lift and
# stack it still scores 0.0%, and would be indistinguishable from the old behaviour by number alone.
#
# So: pull the newest checkpoint off the box, roll out the REACHING family -- the one that requires
# grasping from scratch -- and lay the episodes out as contact sheets to look at.
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
PY=/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python
BOX=${BOX:-b1}
N=${N:-14}

echo "=== pulling the newest checkpoint from $BOX ==="
"$CS/pull_checkpoint.sh" "$BOX" 2>&1 | tail -3
CK=$(ls -t "$CS/results/ck_$BOX"/model_*.pt 2>/dev/null | head -1)
[ -z "${CK:-}" ] && { echo "no checkpoint on $BOX yet"; exit 1; }
echo "checkpoint: $CK"

OUT=$CS/results/video_grasp_$BOX
rm -rf "$OUT"
echo "=== rolling out $N REACHING episodes ==="
"$CS/run.sh" "$CS/render_rollout.py" \
  --task OmniReset-UR10eDelto-CubeStackNoOrient-RelCartesianOSC-State-Play-v0 \
  --checkpoint "$CK" --num_envs 1 --episodes "$N" \
  --reset_type ObjectAnywhereEEAnywhere \
  --collision_stack_mb 1024 --out "$OUT" > "$CS/logs/render_grasp_$BOX.log" 2>&1
grep -E 'RENDER_RESULT' "$CS/logs/render_grasp_$BOX.log" || {
  echo "RENDER FAILED:"; tail -8 "$CS/logs/render_grasp_$BOX.log" | tr '\r' '\n' | tail -6; exit 1; }

# The interesting episodes are the ones where the hand got closest to the cube, because that is where
# closure would show. Ranked by final object-to-goal error only as a proxy -- if the cube moved at
# all, the hand did something to it.
"$PY" - "$OUT" <<'PY'
import json, subprocess, sys, pathlib
d = pathlib.Path(sys.argv[1])
meta = json.loads((d / "episodes.json").read_text())
eps = sorted(meta["episodes"], key=lambda e: e["final_position_error_m"])[:3]
for e in eps:
    out = f"/home/mattephi/Work/Rubitek/cube-stack/report/media/frames/grasp_ep{e['index']:02d}.png"
    subprocess.run([
        "/home/mattephi/Work/Rubitek/uwlab-env/.venv/bin/python",
        "/home/mattephi/Work/Rubitek/cube-stack/contact_sheet.py",
        "--video", str(d / "rollout.mp4"), "--episodes-json", str(d / "episodes.json"),
        "--episode", str(e["index"]), "--frames", "9", "--tile-width", "420", "--out", out],
        check=False)
    hand = e.get("final_hand_to_object_m")
    print(f"  ep{e['index']:>3}  err {e['final_position_error_m']*1000:6.1f} mm  "
          f"hand-to-cube {'-' if hand is None else f'{hand*1000:.0f} mm'}")
PY
"$PY" "$CS/collect_status.py"
echo "=== grasp check done; sheets in report/media/frames ==="
