# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained expert on ONE reset type and report its success rate.

Built to answer a question a synthesized grasp harness could not: does the interior blocker
actually BLOCK the one-sided rim pinch? Rather than trying to reproduce that pinch by hand
(attempts stalled -- the jaws caught before reaching the wall even on the blocker-free v1 jig),
this uses a policy that demonstrably performs it: the v1 expert, whose task_0 grasp IS the
one-sided pinch.

Play that same expert against different jig assets and compare task_0 success:
  * on ``jig``    -> its native asset; the pinch works, so success should be high;
  * on ``jigv2c`` -> if the blocker blocks the pinch, success should COLLAPSE.
A blocker that leaves the number unchanged is not blocking anything.

    ./uwlab.sh -p scripts_v2/tools/conversions/eval_expert_success.py \
      --task OmniReset-UR10eLinearGripper-RelCartesianOSC-State-Finetune-Play-v0 \
      --checkpoint logs/.../model_5600.pt --num_envs 64 --episodes 6 --headless \
      --reset_type ObjectAnywhereEEAnywhere \
      env.scene.insertive_object=jig env.scene.receptive_object=bottomenclosure \
      env.events.reset_from_reset_states.params.dataset_dir=./Datasets_jig/OmniReset
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=6, help="episode-lengths to roll out")
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEAnywhere")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from typing import cast

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

# play.py strips runner-only keys (e.g. 'optimizer') that this rsl_rl PPO does not accept.
sys.path.insert(0, os.path.join(os.getcwd(), "scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    asset = os.path.basename(os.path.dirname(env.scene["insertive_object"].cfg.spawn.usd_path))
    wrapped = RslRlVecEnvWrapper(env)

    ckpt = retrieve_file_path(args_cli.checkpoint)
    sanitized = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    agent_cfg_d = sanitized.to_dict() if hasattr(sanitized, "to_dict") else dict(sanitized)
    runner = OnPolicyRunner(wrapped, agent_cfg_d, log_dir=None, device=env.device)
    # Load the ACTOR only. The CRITIC's input width is privileged-obs dependent and differs
    # between the v1 checkpoint (319) and this env (328); it plays no part in inference, so a
    # strict load would fail for no reason. Assert the actor matched exactly -- silently
    # dropping actor weights would make every number below meaningless.
    sd = torch.load(ckpt, map_location=env.device)["model_state_dict"]
    actor_sd = {k: v for k, v in sd.items() if k.startswith(("actor", "std", "actor_obs_normalizer"))}
    runner.alg.policy.load_state_dict(actor_sd, strict=False)  # rsl_rl overrides this -> bool
    # Verify by value, not by return code: confirm a real actor tensor now equals the file.
    live = dict(runner.alg.policy.named_parameters())
    live.update(dict(runner.alg.policy.named_buffers()))
    checked = 0
    for k, v in actor_sd.items():
        if k in live and live[k].shape == v.shape:
            assert torch.allclose(live[k].float(), v.float().to(live[k].device)), \
                f"actor tensor {k} did not load"
            checked += 1
    assert checked >= 4, f"only {checked} actor tensors verified -- load is suspect"
    print(f"[eval] actor loaded and VERIFIED ({checked}/{len(actor_sd)} tensors match file); critic skipped",
          flush=True)
    runner.alg.policy.eval()
    policy = runner.get_inference_policy(device=env.device)
    print(f"[eval] asset={asset}  reset_type={args_cli.reset_type}  envs={args_cli.num_envs}\n"
          f"       ckpt={ckpt}", flush=True)

    success_fn = env.reward_manager.get_term_cfg("progress_context").func
    # Classify HOW it succeeded, not just whether. A one-sided rim pinch clamps a ~13 mm wall;
    # the two-sided straddle spans the jig's 129 mm. Recording the jaw gap on the successful
    # step separates "blocker stopped the pinch, legitimate grasps still work" from "the pinch
    # still works half the time" -- which the success rate alone cannot do.
    _rob = env.scene["robot"]
    _obj = env.scene["insertive_object"]
    _jaw = _rob.find_joints(["finger_joint", "right_finger_joint"])[0]
    # Sample the gap MID-CARRY, at each env's highest lift. Sampling at episode end is wrong:
    # this is pick-and-place, so by then the gripper has already released (measured: gaps came
    # back 0.8 mm and 136.8 mm -- fully shut and fully open, not a grasp width at all).
    gaps = []
    _peak_z = torch.full((env.num_envs,), -1e9, device=env.device)
    _peak_gap = torch.zeros(env.num_envs, device=env.device)
    ep_len = int(env.max_episode_length)
    n_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    n_succ = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    obs = wrapped.get_observations()   # this wrapper returns a TensorDict, not a tuple
    for _ in range(args_cli.episodes * ep_len):
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        g_now = 136.76 - _rob.data.joint_pos[:, _jaw].sum(dim=1) * 1000.0
        z_now = _obj.data.root_pos_w[:, 2]
        higher = z_now > _peak_z
        _peak_gap = torch.where(higher, g_now, _peak_gap)
        _peak_z = torch.where(higher, z_now, _peak_z)
        d = dones.nonzero(as_tuple=False).flatten()
        if d.numel():
            sc = success_fn.success[d]
            if sc.any():
                gaps.extend(_peak_gap[d][sc].detach().cpu().tolist())
            n_succ[d] += sc.long()
            n_done[d] += 1
            _peak_z[d] = -1e9; _peak_gap[d] = 0.0

    tot_d, tot_s = int(n_done.sum()), int(n_succ.sum())
    rate = tot_s / max(tot_d, 1)
    print(f"\n  asset={asset}  episodes={tot_d}  successes={tot_s}  "
          f"SUCCESS RATE = {rate:.3f}", flush=True)
    if gaps:
        import numpy as _np
        a = _np.array(gaps)
        pinch = int((a < 40).sum()); straddle = int((a > 100).sum())
        print(f"  jaw gap AT PEAK LIFT (mm): median {_np.median(a):.1f}  "
              f"p10 {_np.percentile(a,10):.1f}  p90 {_np.percentile(a,90):.1f}", flush=True)
        print(f"  -> ONE-SIDED PINCH (<40mm): {pinch} ({100*pinch/len(a):.0f}%)   "
              f"STRADDLE (>100mm): {straddle} ({100*straddle/len(a):.0f}%)   "
              f"other: {len(a)-pinch-straddle}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
