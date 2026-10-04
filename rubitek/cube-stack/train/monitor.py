#!/usr/bin/env python
"""One status line for the newest DELTO pinch training run (or --run <dir>).

Reads the run's TensorBoard events and prints: iteration, total steps/s, mean reward, mean episode
length, the end-of-episode success rate and the per-reset-family success rates, each as the latest
value and the best seen. Writes the same line to <run>/status.txt so a wake-up can read it cheaply.
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default="/home/mattephi/Work/Rubitek/UWLab-cubestack/logs/rsl_rl/ur10e_delto_omnireset_agent")
parser.add_argument("--run", default=None)
args = parser.parse_args()

run = args.run or max((d for d in glob.glob(os.path.join(args.runs, "*")) if os.path.isdir(d)), key=os.path.getmtime)
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
tags = set(ea.Tags()["scalars"])


def last_best(tag):
    if tag not in tags:
        return None
    s = ea.Scalars(tag)
    return s[-1].step, s[-1].value, max(x.value for x in s)


parts = [os.path.basename(run)]
it = last_best("Train/mean_reward")
if it:
    parts.append(f"iter {it[0]}")
    parts.append(f"reward {it[1]:.2f} (best {it[2]:.2f})")
for tag, label in (("Perf/total_fps", "fps"), ("Train/mean_episode_length", "ep_len"),
                   ("Metrics/task_command/end_of_episode_success_rate", "success")):
    v = last_best(tag)
    if v:
        parts.append(f"{label} {v[1]:.3f}" + (f" (best {v[2]:.3f})" if label == "success" else ""))
fam = []
for i in range(4):
    v = last_best(f"Metrics/task_{i}_success_rate")
    if v:
        fam.append(f"{v[1]:.2f}")
if fam:
    parts.append("per-family [" + " ".join(fam) + "]")
line = " | ".join(parts)
print(line)
with open(os.path.join(run, "status.txt"), "w") as f:
    f.write(line + "\n")
