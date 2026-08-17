# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Options A/B pilot (beads UWLab-dwx.6, dwx.7): measure sample efficiency for spawning the leg
NEAR (A) vs AT (B) the assembled target pose next to the fixture, then letting the policy grasp it
in place, held-check verified.

THE SCENE/CONTROL-RATE CONFLICT, AND HOW IT IS RESOLVED HERE: the fixture only exists in the
OmniReset UR5eDelto family (variant pair leg200mm + onelegfixture), which runs at 10 Hz
(decimation=12). The certified Stage-2 lift checkpoint trained at 60 Hz (decimation=2) in dexlift's
own env, which has no fixture. Stepping the 60 Hz checkpoint inside the 10 Hz OmniReset env would
silently feed it observations/require actions at 6x the control-rate spacing it was trained under --
exactly the trap flagged after UWLab-dwx.2. Resolution taken here: ADD THE FIXTURE TO THE DEXLIFT
ENV AT ITS NATIVE 60 Hz, rather than re-timing anything or touching the OmniReset family. This is a
pure scene addition (env_cfg.scene.fixture = a plain kinematic RigidObjectCfg, added the same way
the generator script adds terminations.success -- a dynamically-set instance attribute that
IsaacLab's SceneCfg discovers the same way TerminationsCfg does, confirmed already for terminations
in dwx.2/dwx.8) plus one new reset-mode event that places the leg relative to the fixture using the
SAME assembled_offset metadata composition MAIN's events.py:945-980 uses -- read live from each
USD's metadata.yaml, not hardcoded, per this job's constraint on the leg's mass (being edited live
by another agent) generalised to "read metadata live, do not hardcode any of it".

Nothing about the dexlift env's own decimation/sim.dt/action space/termination set is touched
beyond adding the fixture asset and the one placement event; the plant the checkpoint was certified
on is otherwise unmodified.

OPTION A: spawn the leg near the assembled/inserted pose, with a small random position/orientation
perturbation (mirrors upstream's own assembly_sampling_event, which the user should be told already
does this -- upstream's "partially assembled" is NOT a partially-screwed leg, it is the exact
assembled pose plus tiny collision-free force/torque perturbation).
OPTION B: spawn the leg EXACTLY at the assembled target pose, zero perturbation. Because the
held-check's probe requires the object to demonstrably track a jog of the grasping hand, a leg that
starts already "in position" can only register as accepted if the hand genuinely grasps it -- the
probe is precisely what makes this option meaningful rather than trivially satisfied by a leg that
never needed to move.

Run (one Isaac process per option; never via uwlab.sh):
    PYTHONPATH=... timeout -s KILL 600 <python> -u scripts_v2/tools/pilot_options_ab.py \\
        --option A --checkpoint <path>.pth --num_envs 64 --num_episodes 200
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Lift-Play-v0")
parser.add_argument("--option", type=str, choices=["A", "B"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_yaml", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200, help="Stop once roughly this many episodes have ended.")
parser.add_argument("--max_steps", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import yaml  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.dexlift.mdp as dexlift_mdp  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_LOCAL_ASSETS_DIR  # noqa: E402
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.held_check_core import PROBE_ARM_ACTION_BIAS  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import read_metadata_from_usd_directory  # noqa: E402

FIXTURE_USD = f"{UWLAB_ASSETS_DATA_DIR}/Props/FurnitureBench/OneLegInsertionFixture/one_leg_insertion_fixture.usd"
LEG_USD = f"{UWLAB_LOCAL_ASSETS_DIR}/Props/FurnitureBench/SquareTableLeg200mmDecomp/square_table_leg4_200mm.usd"

# Fixed fixture placement: near the leg's own workspace centre (WORKSPACE_X=0.55, table.py) but
# offset in y so the leg's default spawn footprint and the fixture do not initially overlap.
FIXTURE_POS = (0.55, 0.20, 0.10)

# Option A perturbation: mirrors upstream's assembly_sampling_event (tiny force/torque-scale
# jitter that stays collision-free) as a small pos/rot jitter instead, since we are writing the
# pose directly rather than running a force-perturbation sub-episode. +-8mm / +-8deg is
# deliberately small -- "very near the final inserted pose", not a fresh random placement.
OPTION_A_POS_JITTER_M = 0.008
OPTION_A_ROT_JITTER_RAD = 0.14  # ~8 deg


def unwrap(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


def main() -> None:
    agent_yaml = args_cli.agent_yaml or os.path.join(
        os.path.dirname(os.path.dirname(args_cli.checkpoint)), "params", "agent.yaml"
    )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.sim.physx.gpu_collision_stack_size = 2**24
    env_cfg.seed = None

    # -- SCENE: add the fixture (kinematic, fixed pose -- see FIXTURE_POS comment above).
    env_cfg.scene.fixture = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FIXTURE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=FIXTURE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # -- HELD-CHECK, same wiring as the main generator.
    env_cfg.terminations.success = DoneTerm(func=dexlift_mdp.held_with_probe, time_out=True, params={})

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[pilot] option={args_cli.option} fixture pos={FIXTURE_POS}", flush=True)

    # -- assembled-offset composition, read live from each USD's own metadata.yaml (P5: reader
    # composes insertive onto receptive; both offsets carry quat=[1,0,0,0] here, so this collapses
    # to translation-only -- see the module docstring for the full derivation).
    leg_meta = read_metadata_from_usd_directory(LEG_USD)
    fixture_meta = read_metadata_from_usd_directory(FIXTURE_USD)
    leg_offset = torch.tensor(leg_meta["assembled_offset"]["pos"], device=env.device, dtype=torch.float32)
    fixture_offset = torch.tensor(fixture_meta["assembled_offset"]["pos"], device=env.device, dtype=torch.float32)
    fixture_pos_w = torch.tensor(FIXTURE_POS, device=env.device, dtype=torch.float32) + env.scene.env_origins
    target_point_w = fixture_pos_w + fixture_offset  # fixture assembled_offset assumed identity quat
    leg_target_pos_w = target_point_w - leg_offset  # leg assembled_offset assumed identity quat
    leg_target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    print(f"[pilot] leg target pos (env 0, world): {leg_target_pos_w[0].tolist()}", flush=True)

    def place_leg(env_ids: torch.Tensor) -> None:
        # NOTE: callers whose env_ids/context originates from inside a `torch.inference_mode()`
        # block (i.e. every call from within the rollout loop, where `done_idx` comes from a
        # tensor computed during a policy step) must invoke this from INSIDE that same block --
        # PyTorch refuses an in-place write onto simulation buffers using inference-mode-sourced
        # data from OUTSIDE inference_mode. Wrapping inference_mode HERE instead, tried first,
        # poisoned the asset's internal pose/velocity buffers for the rest of the process (a LATER,
        # unrelated write inside the environment's own reset_root_state_uniform then failed with
        # the same error) -- inference_mode has to bracket the call site, not be re-entered locally.
        n = env_ids.numel()
        pos = leg_target_pos_w[env_ids].clone()
        quat = leg_target_quat_w[env_ids].clone()
        if args_cli.option == "A":
            pos = pos + (torch.rand(n, 3, device=env.device) * 2 - 1) * OPTION_A_POS_JITTER_M
            rpy = (torch.rand(n, 3, device=env.device) * 2 - 1) * OPTION_A_ROT_JITTER_RAD
            delta_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            quat = math_utils.quat_mul(delta_quat, quat)
        leg = env.scene["object"]
        leg.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        leg.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    # place at construction-time reset too
    place_leg(torch.arange(env.num_envs, device=env.device))

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    wrapped_env = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_act, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.has_batch_dimension = True
    print("[pilot] POLICY_LOADED", flush=True)

    obs = unwrap(wrapped_env.reset())
    place_leg(torch.arange(env.num_envs, device=env.device))  # env.reset() re-samples via reset_object; overwrite

    success_term = env.termination_manager.get_term_cfg("success").func
    n_attempts = 0
    gate_names = ["settled", "opposed_contact", "co_move", "probe_ready", "probe_gripper_moved", "probe_tracks"]
    rejection_counts = {g: 0 for g in gate_names}
    rejection_counts["accepted"] = 0

    for _ in range(args_cli.max_steps):
        with torch.inference_mode():
            act = player.get_action(obs, is_deterministic=True)
            in_probe = success_term.probe_active
            if in_probe.any():
                act[in_probe, 0:6] = act[in_probe, 0:6] + PROBE_ARM_ACTION_BIAS
            ret = wrapped_env.step(act)
            obs = unwrap(ret[0])
            dones = ret[2].flatten().to(torch.bool)

            if dones.any():
                breakdown = success_term.gate_breakdown(env)
                success_now = env.termination_manager.get_term("success")
                done_idx = torch.nonzero(dones).flatten()
                n_attempts += done_idx.numel()
                for idx in done_idx.tolist():
                    if bool(success_now[idx]):
                        rejection_counts["accepted"] += 1
                        continue
                    for g in gate_names:
                        if not bool(breakdown[g][idx]):
                            rejection_counts[g] += 1
                            break
                # re-place the leg for the envs that just reset (env.step()'s auto-reset already
                # ran reset_object with its own random sampling; overwrite with A/B placement).
                # Must stay INSIDE this inference_mode block -- see place_leg's own docstring.
                place_leg(done_idx)

        if n_attempts >= args_cli.num_episodes:
            break
        if env.sim.is_stopped():
            break

    print(f"\n=== OPTION {args_cli.option} RESULT ===", flush=True)
    print(f"attempts (episodes ended): {n_attempts}", flush=True)
    print(f"accepted: {rejection_counts['accepted']}", flush=True)
    if n_attempts > 0:
        print(f"efficiency: {rejection_counts['accepted']/n_attempts:.2%}", flush=True)
    print(f"probes armed: {success_term.n_probes_armed}  finalized: {success_term.n_probes_finalized}  "
          f"tracked: {success_term.n_probes_tracked}", flush=True)
    print("rejection breakdown (first failing gate):", flush=True)
    for g in gate_names + ["accepted"]:
        print(f"  {g:22s}: {rejection_counts[g]}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
