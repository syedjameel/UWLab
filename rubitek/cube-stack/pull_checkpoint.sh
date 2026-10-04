#!/usr/bin/env bash
# Fetch one box's newest checkpoint, for evaluation or rendering on the local card.
#
# Rendering has to happen here regardless: the deployment guide requires an RTX-class GPU on driver
# branch 580 for any camera-enabled run, and the rented A6000 is on 570 and is used headless only.
#
#   ./pull_checkpoint.sh b1        newest checkpoint from box 1
#   ./pull_checkpoint.sh b2 model_150.pt
set -uo pipefail
CS=/home/mattephi/Work/Rubitek/cube-stack
KEY=/home/mattephi/Work/IRiS/ip/meaningful-latents/.secrets/id_ed25519_ml_vast
BOX=${1:?usage: pull_checkpoint.sh <b1|b2> [model_N.pt]}
WANT=${2:-}

case "$BOX" in
  b1) PORT=40624; HOST=root@38.29.145.10 ;;
  b5) PORT=45350; HOST=root@115.73.217.180 ;;
  b3) PORT=17626; HOST=root@89.221.67.158 ;;
  b4) PORT=40523; HOST=root@38.117.87.51 ;;
  *)  echo "unknown box $BOX"; exit 1 ;;
esac
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT"
REMOTE=/workspace/UWLab/logs/rsl_rl/ur10e_delto_omnireset_agent

if [ -z "$WANT" ]; then
  SRC=$($SSH "$HOST" "find $REMOTE -name 'model_*.pt' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-")
else
  SRC=$($SSH "$HOST" "find $REMOTE -name '$WANT' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-")
fi
[ -z "${SRC:-}" ] && { echo "no checkpoint found on $BOX"; exit 1; }

DST=$CS/results/ck_$BOX
mkdir -p "$DST"
rsync -az --progress -e "$SSH" "$HOST:$SRC" "$DST/" || exit 1
# The params directory records the exact env and agent config the checkpoint was trained under, so
# an evaluation can be checked against it rather than assumed to match.
rsync -az -e "$SSH" "$HOST:$(dirname "$SRC")/params/" "$DST/params/" 2>/dev/null

echo
echo "pulled  $SRC"
ls -la "$DST"/*.pt
echo "$SRC" > "$DST/SOURCE"
