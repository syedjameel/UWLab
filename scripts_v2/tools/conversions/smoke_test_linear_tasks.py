# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Smoke-test the linear-gripper task wiring: build each env (num_envs=1) and, where the task
is self-contained (no external dataset), step it. Reports OK/FAIL per task without needing the
reset-states / grasp datasets. Catches joint-regex, action, robot-load and obs wiring errors.

    ./uwlab.sh -p scripts_v2/tools/conversions/smoke_test_linear_tasks.py
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Single task id to build (one per process).")
parser.add_argument("--step", action="store_true", help="Also reset+step the env (self-contained tasks only).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys
import traceback

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import uwlab_tasks  # noqa: F401

# IMPORTANT: build exactly ONE env per process. Closing a GPU env and creating another in the
# same process corrupts the PhysX GPU context (computeArticulationData CUDA error 2), so the
# caller loops over tasks with a fresh process each (see run_smoke_all.sh).
try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg).unwrapped
    if args.step:
        env.reset()
        a = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
        for _ in range(5):
            env.step(a)
        verdict = "OK (built + stepped)"
    else:
        verdict = "OK (built)"
    print(f"[SMOKE_RESULT] [PASS] {args.task}: {verdict}")
except Exception as e:  # noqa: BLE001
    print(f"[SMOKE_RESULT] [FAIL] {args.task}: {type(e).__name__}: {e}")
    traceback.print_exc()

sys.stdout.flush()
os._exit(0)
