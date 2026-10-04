# Compute & budget plan

## Accounts / access (verified 2026-08-31)
vast.ai **Sinew team**, user id 636218, `is_team=true`, credit **$65.76**, balance $0, 0 instances running.
All calls go through `~/Work/IRiS/ip/meaningful-latents/scripts/vast` which passes `--api-key`
from the project-local `.secrets/vast_api_key` and refuses `set api-key`. The global
`~/.config/vastai/vast_api_key` is a DIFFERENT account (ooju) and is never touched — key hashes
differ (`140374af…` vs `8bf83464…`).

## Machines
| role | machine | why |
|---|---|---|
| dev, debug, reset-bank generation, **all rendering/video** | local RTX 4070 Ti Super 16 GB, driver 580.159.04 | free; guide requires RTX-class + driver branch 580 for rendering (595 segfaults `rtx.scenedb`; A100/H100 cannot render at all) |
| PPO training | vast, headless | paid, no rendering needed |

## Vast pricing surveyed 2026-08-31
| config | $/h | 24 h |
|---|---|---|
| 4x RTX 3090 24 GB | **0.588** ($0.147/gpu) | $14.1 |
| 4x Q RTX 6000 22 GB | 0.534 | $12.8 |
| 1x RTX 3090 | 0.148 | $3.6 |
| 1x A40 / 4090 46-49 GB | 0.34-0.38 | $8-9 |

`--distributed` is supported by `scripts/reinforcement_learning/rsl_rl/train.py:31` (torchrun,
sets `cuda:{local_rank}` and offsets the seed per rank), so multi-GPU is a flag, not a port.

## Budget envelope (of $65.76)
- reserve, untouchable: **$20**
- variant B (no orientation, primary): ~24 h on 4x 3090 = **$14**
- variant A (with orientation): ~24 h on 4x 3090 = **$14**
- bring-up, smoke tests, bank generation, evals, re-runs: **$15**
Rule: never leave an instance idle; destroy on completion; check `show instances` every wake-up.

## Paper's own compute, for calibration
Peg 8 h / Leg 16 h / Drawer 24 h, each on 4x L40S. Cube: not stated. 3090 is ~2-2.5x slower than
L40S for this workload, but our task is simpler than Leg/Drawer and we drop the sim2real dynamics
curriculum entirely (the paper's sim-only numbers are trained without it, §5/§A.3.9).
