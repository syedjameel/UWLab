#!/usr/bin/env bash
# Pre-staged curriculum job. Dropped into ~/rubitek/jobs/ BEFORE provisioning finishes: the job loop
# only arms after the provision script writes READY, so this sits untouched until the box is actually
# ready and then starts instantly, instead of idling until I next look at it.
#
# The effort scale is baked in rather than left to the script's default. It comes from the endurance
# sweep measured on the local 4070 Ti (C3 bank, null action, 128 envs, in_hand at step 150):
#
#     x1  (shipped) 0.047      x3   0.336      x10  0.523      x20  0.586      x40  0.625
#
# x20 is the top of the flat region: x40 gains 0.04, which is inside the 0.062-0.180 run-to-run noise
# band, while pushing fingertip force past the DG-5F's ~68 N rated grip. Tucking fingers 3/4/5 is NOT
# enabled -- on its own it made retention worse (0.023 vs 0.047) and added nothing at raised effort.
set -uo pipefail
WS=${WS:-$HOME/rubitek}
export UWLAB_HAND_EFFORT_SCALE=20
export UWLAB_TUCK_OTHERS=0
export UWLAB_GRIP_BIAS_RAD=0.05
# 2048, not 4096. MEASURED: at 4096 envs this box did 4 iterations in 353 s = 0.68 it/min, which puts
# the original 150+600 budget at ~18 hours and stage A alone past 04:50. Throughput is
# simulation-bound and roughly fixed in env-steps/s (~1,100/s here, only 1.4x the local 4070 Ti's
# ~780/s, since GPU 0 carries another user), so halving the env count roughly doubles it/min for the
# same data volume -- more PPO updates and more checkpoints inside the hours that remain.
export N=${N:-2048}
export NAME=h100_eff20
# TIME-BOXED, not chosen for completeness. The local box measures 1.67 iterations/min at 1024 envs;
# even assuming the H100 is ~1.6x faster at 4096 envs, the original 250+900 budget is ~7.7 hours and
# would finish around 08:30 -- after the morning deadline, with no slack left to render films or
# build the report. 150+600 lands near 05:30 and leaves that slack.
#
# The split favours stage B deliberately. Stage A (C4 only) already reached 0.653 from scratch in
# ~460 iterations on the OLD broken dynamics, so it is the part that is known to work; C3 is the
# family that has never once scored above 0.0000 and is the entire point of the night, so it gets
# the hours. Checkpoints land every 25 iterations, so an early stop still leaves something to film.
# Re-budgeted against the measured rate, not against hope. At ~1.4 it/min (2048 envs) this is about
# 4.8 hours, finishing near 06:10 -- inside the morning deadline with room to build the report.
# Stage B keeps the lion's share deliberately: C3 is the family that has never scored above 0.0000
# and is the entire point of the night, while stage A only has to bootstrap a grasp that already
# works. Checkpoints land every 25 iterations, so an early stop still leaves something usable.
export IT_A=80 IT_B=320 IT_C=0 IT_D=0
exec bash "$WS/train_h100.sh"
