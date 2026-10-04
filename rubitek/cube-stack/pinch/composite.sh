#!/usr/bin/env bash
# Stack per-posture clips side by side with a caption burned into each panel.
#
# Each panel is rendered as its OWN single-environment run so the camera can frame that jaw
# tightly; a single wide shot of three environments puts the interesting 30 mm of geometry
# across about 40 pixels, which is how the first attempt ended up showing a mount and no jaw.
#
# Usage: composite.sh OUT.mp4 IN1.mp4 "Label 1" IN2.mp4 "Label 2" [IN3.mp4 "Label 3" ...]
set -euo pipefail

OUT="$1"; shift
FONT=/usr/share/fonts/TTF/DejaVuSans-Bold.ttf
[ -f "$FONT" ] || FONT=$(fc-match -f '%{file}' "DejaVu Sans:bold" 2>/dev/null || echo "")

inputs=(); filters=(); labels=(); n=0
while [ $# -gt 0 ]; do
  src="$1"; txt="$2"; shift 2
  [ -f "$src" ] || { echo "composite: missing $src, skipping" >&2; continue; }
  inputs+=(-i "$src")
  esc=$(printf '%s' "$txt" | sed "s/[:']/\\\\&/g")
  if [ -n "$FONT" ]; then
    filters+=("[${n}:v]scale=640:-2,pad=iw:ih+58:0:0:color=0x111111,drawtext=fontfile=${FONT}:text='${esc}':x=(w-text_w)/2:y=h-42:fontsize=21:fontcolor=white[v${n}]")
  else
    filters+=("[${n}:v]scale=640:-2[v${n}]")
  fi
  labels+=("[v${n}]")
  n=$((n+1))
done

[ "$n" -gt 0 ] || { echo "composite: no inputs" >&2; exit 1; }

if [ "$n" -eq 1 ]; then
  chain="${filters[0]};${labels[0]}copy[out]"
else
  chain="$(IFS=';'; echo "${filters[*]}");$(IFS=''; echo "${labels[*]}")hstack=inputs=${n}[out]"
fi

ffmpeg -y -loglevel error "${inputs[@]}" -filter_complex "$chain" \
  -map '[out]' -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT"
echo "WROTE $OUT"
