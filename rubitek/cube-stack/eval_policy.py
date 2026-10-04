"""Measure a trained cube-stacking policy's success rate, honestly.

WHAT THIS REPORTS, AND WHY BOTH NUMBERS
---------------------------------------
Two success rates, always, side by side:

  held-at-end   the success predicate is satisfied on the FINAL step of the episode
  ever-during   the predicate was satisfied at ANY step, even if the cube was dropped afterwards

They are not interchangeable and the gap between them is informative. This branch's
`certify_pose.py` grades with a sticky-OR ("an episode counts once the predicate holds at ANY step,
even if the object is dropped afterwards"), which flatters a policy that touches the goal and then
knocks the cube off. Reporting only that number would be misleading; reporting only held-at-end
would undersell a policy whose cube settles a millimetre out on the last frame. So: both, labelled.

The headline number is measured from the REACHING distribution -- the Play configs reset from
`ObjectAnywhereEEAnywhere` alone (`rl_state_cfg.py`), which is the full-task distribution and the
one the paper reports. Near-goal-only success is easy and is not the claim.

Episodes are counted per-environment and only COMPLETED episodes are tallied. A partially-elapsed
episode at the end of the run is discarded rather than counted as a failure.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="completed episodes to tally")
parser.add_argument("--insertive", type=str, default="cube34")
parser.add_argument("--receptive", type=str, default="cube34")
parser.add_argument("--dataset_dir", type=str, default="./Datasets_ur10e_delto/OmniReset")
parser.add_argument("--collision_stack_mb", type=int, default=None)
parser.add_argument("--solver_iters", type=int, default=None,
                    help="Override solver_position_iteration_count (default 36). Lowering it is the\n"
                         "cheap half of the physics-cost question, but every reset bank was recorded\n"
                         "at 36, so a lower count replays those states under different contact\n"
                         "physics. Grading the SAME checkpoint at each value is what tells a cost\n"
                         "saving apart from a change to the task.")
parser.add_argument("--action_noise", type=float, default=0.0,
                    help="std of Gaussian noise added to actions. The paper's OFFLINE SEED-SELECTION "
                         "PROXY (§A.3.10): evaluate each seed under injected action noise and take "
                         "the best, because it correlates with real-world robustness. Magnitude is "
                         "not stated in the paper, so sweep it rather than trusting one value.")
parser.add_argument("--motion_stiffness", type=float, default=None,
                    help="Override the OSC's TRANSLATIONAL stiffness, to match how the policy was "
                         "TRAINED.\n\n"
                         "The curricula pass this as a hydra override, so a checkpoint from them was "
                         "fitted against a plant the Play config does not reproduce. Grading it on "
                         "the shipped gains measures a policy on a different robot from the one it "
                         "learned -- the same class of mismatch that would have invalidated every "
                         "eval number in this project had the Play configs used the Finetune action "
                         "group. Pass the training value here.")
parser.add_argument("--motion_damping", type=float, default=None,
                    help="Override the OSC's TRANSLATIONAL damping ratio. See --motion_stiffness.")
parser.add_argument("--out", type=str, default=None, help="write results as JSON here")
parser.add_argument(
    "--reset_type",
    type=str,
    default="ObjectAnywhereEEAnywhere",
    choices=["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
             "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
    help="Which reset family to evaluate FROM. The paper does not report one number either: Fig. 8 "
         "decomposes success into Near-Goal, Grasped, Near-Object and Entire Task, ordered 'roughly "
         "from the end of the task to the beginning'. ObjectAnywhereEEAnywhere is the full task and "
         "is the headline; the others show how far competence has propagated backwards from the "
         "goal, which is the whole mechanism OmniReset claims.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# The repo's own train.py/play.py run the agent cfg through cli_args.sanitize_rsl_rl_cfg before
# handing it to the runner; without it the runner forwards keys the fork's PPO does not accept and
# dies with "PPO.__init__() got an unexpected keyword argument 'optimizer'". Reuse the repo's
# sanitizer rather than reimplementing which keys are legal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/mattephi/Work/Rubitek/UWLab-cubestack/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
v = env_cfg.variants
env_cfg.scene.insertive_object = v["scene.insertive_object"][args.insertive]
env_cfg.scene.receptive_object = v["scene.receptive_object"][args.receptive]
for _, term in vars(env_cfg.events).items():
    p = getattr(term, "params", None)
    if isinstance(p, dict) and "dataset_dir" in p:
        p["dataset_dir"] = args.dataset_dir
if args.collision_stack_mb is not None:
    env_cfg.sim.physx.gpu_collision_stack_size = args.collision_stack_mb * 1024 * 1024
if args.solver_iters is not None:
    props = env_cfg.scene.robot.spawn.articulation_props
    env_cfg.scene.robot.spawn.articulation_props = props.replace(
        solver_position_iteration_count=args.solver_iters
    )
    print(f"[eval] solver_position_iteration_count overridden to {args.solver_iters}")

# Point the reset manager at a single family. The Play cfg ships reset_types=["ObjectAnywhereEEAnywhere"]
# with probs=[1.0]; swapping the name evaluates from a different point in the task.
_rst = getattr(env_cfg.events, "reset_from_reset_states", None)
if _rst is not None:
    _rst.params["reset_types"] = [args.reset_type]
    _rst.params["probs"] = [1.0]

if args.motion_stiffness is not None or args.motion_damping is not None:
    _arm = env_cfg.actions.arm
    _k, _d = list(_arm.motion_stiffness), list(_arm.motion_damping_ratio)
    if args.motion_stiffness is not None:
        _k[:3] = [args.motion_stiffness] * 3
    if args.motion_damping is not None:
        _d[:3] = [args.motion_damping] * 3
    env_cfg.actions.arm = _arm.replace(motion_stiffness=tuple(_k), motion_damping_ratio=tuple(_d))
    print(f"[eval] OSC translational gains -> stiffness {_k[:3]}, damping ratio {_d[:3]}")

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
dev = u.device
N = args.num_envs

# ProgressContext is the single source of truth for the success predicate -- the same object the
# reward reads. Grading off it, rather than recomputing the geometry here, means a pass in this
# script means exactly what a pass during training meant.
ctx = u.reward_manager.get_term_cfg("progress_context").func
cmd = u.command_manager.get_term("task_command")
print(f"[eval] thresholds: position={cmd.success_position_threshold} "
      f"orientation={cmd.success_orientation_threshold}")

ever = torch.zeros(N, dtype=torch.bool, device=dev)
done_end, done_ever, n_done = 0, 0, 0
pos_err_at_end, rot_err_at_end = [], []
# Action magnitude separates "actively places the cube" from "learned to hold still".
# Near-goal resets spawn the cube dislodged from the goal (median 13.5 mm out), so a policy that
# froze would score only the fraction that happened to spawn inside the 5 mm threshold -- which is
# essentially what the untrained baseline measures. A trained policy scoring HIGHER while moving
# NOTHING would be a red flag; scoring higher while moving is partial competence.
act_abs_sum, act_steps = 0.0, 0
arm_abs_sum, hand_abs_sum = 0.0, 0.0

# EPISODE LENGTH IS THE PRE-SOLVED DISCRIMINATOR.
# The termination rule is consecutive_success_state_with_min_length(num_consecutive_successes=5,
# min_episode_length=10), and terminations are computed BEFORE rewards in the step loop, so the check
# at step 10 reads the counter written at step 9. It therefore fires when the predicate held on steps
# 5 through 9 -- the cube was inside tolerance within half a second of the reset, with at most four
# control steps to get there. An earlier run scored 3/43 "successes" of
# which every one ran exactly 10 steps -- resets that spawned inside the 5 mm tolerance, not stacks
# the policy performed. Splitting the success rate by episode length is therefore not a nicety:
# without it the headline number cannot be told apart from a property of the reset distribution.
ep_len = torch.zeros(N, dtype=torch.long, device=dev)
lens_success, lens_fail = [], []

# WHERE THE CUBE ENDS UP, VERTICALLY. Success rate cannot tell "carried and mis-placed" from
# "dropped on the table thirty steps in", and on the grasped families that is the whole question --
# a policy that keeps hold of the cube for the episode has learned something a policy that sheds it
# has not, even when both score zero. The table top puts a 34 mm cube's centre at 0.021 m and the
# base cube puts it at 0.055 m, so the final height separates the two cases directly.
_obj = u.scene["insertive_object"]
TABLE_Z_M = 0.021
ON_TABLE_TOL_M = 0.004
final_z = []

# PROGRESS, NOT POSITION. A final value alone cannot say what the policy DID -- three times over this
# investigation an endpoint was read as an outcome when it was the starting condition:
#   * "cube moved 130 mm" was a displacement magnitude that counted a cube FALLING to the table
#   * a final height distribution (median 54 mm, p90 253 mm) turned out to match the reset bank's
#     STARTING height distribution exactly, so a cube "held high at 191 mm" had been handed to the
#     policy at 191 mm
# Capturing the initial state per episode and reporting the DELTA makes that impossible: a policy
# that changes nothing scores zero progress however extreme its endpoints look.
# CAPTURED AFTER THE FIRST STEP, NOT BEFORE IT. `ProgressContext` fills `xyz_distance` during reward
# evaluation, so before any step has run the tensor is still zeros -- and differencing against zero
# reports `-final_distance` dressed up as "progress", which is exactly the kind of endpoint-read this
# metric exists to prevent. One control step of drift is a negligible price for a reading that means
# what it says.
init_dist = None
init_z = _obj.data.root_pos_w[:, 2].clone()
progress_m, dz_m = [], []

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

with torch.inference_mode():
    while n_done < args.episodes:
        act = policy(obs)
        if args.action_noise > 0:
            act = act + torch.randn_like(act) * args.action_noise
        act_abs_sum += float(act.abs().mean())
        arm_abs_sum += float(act[:, :6].abs().mean())
        hand_abs_sum += float(act[:, 6:].abs().mean())
        act_steps += 1
        obs, _, dones, _ = env.step(act)
        if isinstance(obs, tuple):
            obs = obs[0]
        ep_len += 1

        if init_dist is None:
            init_dist = ctx.xyz_distance.clone()
            init_z = _obj.data.root_pos_w[:, 2].clone()

        succ = ctx.success.clone()
        ever |= succ

        d = dones.bool().flatten()
        if d.any():
            k = int(d.sum())
            # Read the predicate BEFORE the auto-reset overwrites it: ManagerBasedRLEnv resets a
            # done env inside the same step() call, so anything read later describes a fresh episode.
            done_end += int(succ[d].sum())
            done_ever += int(ever[d].sum())
            pos_err_at_end += ctx.xyz_distance[d].tolist()
            rot_err_at_end += ctx.euler_xy_distance[d].tolist()
            sd = d & succ
            fd = d & ~succ
            lens_success += ep_len[sd].tolist()
            lens_fail += ep_len[fd].tolist()
            # Read before the auto-reset overwrites it, exactly as the predicate above is.
            final_z += _obj.data.root_pos_w[d, 2].tolist()
            # ctx.xyz_distance was read into `succ`/`pos_err_at_end` above, BEFORE the auto-reset
            # overwrote it; the same applies here, so compare against this episode's own start.
            progress_m += (init_dist[d] - ctx.xyz_distance[d]).tolist()
            dz_m += (_obj.data.root_pos_w[d, 2] - init_z[d]).tolist()
            # The env has already reset these slots inside step(), so what is readable now is the
            # NEXT episode's initial condition. Re-capture it for them and leave the others alone.
            init_dist[d] = ctx.xyz_distance[d]
            init_z[d] = _obj.data.root_pos_w[d, 2]
            n_done += k
            ever[d] = False
            ep_len[d] = 0

env.close()


def pct(a, b):
    return 100.0 * a / max(1, b)


def q(xs, p):
    t = torch.tensor(xs)
    return float(t.quantile(p)) if len(xs) else float("nan")


res = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "episodes": n_done,
    "action_noise": args.action_noise,
    "solver_iters": args.solver_iters,
    "motion_stiffness": args.motion_stiffness,
    "motion_damping": args.motion_damping,
    "num_envs": args.num_envs,
    "reset_distribution": args.reset_type,
    "success_held_at_end_pct": pct(done_end, n_done),
    "success_ever_during_pct": pct(done_ever, n_done),
    # HOW CLOSE DOES THE BEST EPISODE GET? Median and p90 describe the bulk, which on Stable-Grasp
    # just tracks the reset distribution and says nothing about whether the 5 mm ball is within
    # reach. The lower tail is what decides whether the remaining gap is a precision problem worth
    # shaping for, or something structural that no amount of shaping closes.
    "position_error_m": {
        "min": (min(pos_err_at_end) if pos_err_at_end else float("nan")),
        "p10": q(pos_err_at_end, 0.1),
        "median": q(pos_err_at_end, 0.5),
        "p90": q(pos_err_at_end, 0.9),
        "frac_within_20mm": (sum(1 for x in pos_err_at_end if x < 0.020) / len(pos_err_at_end)
                             if pos_err_at_end else float("nan")),
        "frac_within_50mm": (sum(1 for x in pos_err_at_end if x < 0.050) / len(pos_err_at_end)
                             if pos_err_at_end else float("nan")),
    },
    "euler_xy_error_rad": {"median": q(rot_err_at_end, 0.5), "p90": q(rot_err_at_end, 0.9)},
    "thresholds": {
        "position_m": cmd.success_position_threshold,
        "euler_xy_rad": cmd.success_orientation_threshold,
    },
    "episode_length_success": {
        "n": len(lens_success),
        "min": min(lens_success) if lens_success else None,
        "median": (sorted(lens_success)[len(lens_success) // 2] if lens_success else None),
        "max": max(lens_success) if lens_success else None,
        "n_exactly_10": sum(1 for x in lens_success if x == 10),
        "n_gt_10": sum(1 for x in lens_success if x > 10),
        # Counted from the raw lengths, not derived from the histogram below. The histogram's
        # buckets are 10 wide, so "successes longer than 20 steps" read off it is really ">= 20" --
        # it cannot express the stated definition, and quietly reported an upper bound.
        "n_gt_20": sum(1 for x in lens_success if x > 20),
        "histogram": {str(b): sum(1 for x in lens_success if b <= x < b + 10) for b in range(0, 210, 10)},
    },
    "episode_length_fail": {
        "n": len(lens_fail),
        "median": (sorted(lens_fail)[len(lens_fail) // 2] if lens_fail else None),
    },
    "goal_progress_m": {
        "median": q(progress_m, 0.5),
        "p10": q(progress_m, 0.1),
        "p90": q(progress_m, 0.9),
        "frac_closer": (sum(1 for x in progress_m if x > 0.005) / len(progress_m)
                        if progress_m else float("nan")),
        "n": len(progress_m),
    },
    "object_dz_m": {"median": q(dz_m, 0.5), "p10": q(dz_m, 0.1), "p90": q(dz_m, 0.9)},
    "final_object_z_m": {
        "median": q(final_z, 0.5),
        "p10": q(final_z, 0.1),
        "p90": q(final_z, 0.9),
        "frac_on_table": (
            sum(1 for z in final_z if abs(z - TABLE_Z_M) < ON_TABLE_TOL_M) / len(final_z)
            if final_z else float("nan")
        ),
        "n": len(final_z),
    },
    "mean_abs_action": act_abs_sum / max(1, act_steps),
    "mean_abs_action_arm": arm_abs_sum / max(1, act_steps),
    "mean_abs_action_hand": hand_abs_sum / max(1, act_steps),
}
print("\n" + json.dumps(res, indent=2))
_ge = res["episode_length_success"]
_gen = _ge["n_gt_10"]
res["genuine_success_pct"] = 100.0 * _gen / max(1, n_done)
res["sustained_success_pct"] = 100.0 * _ge["n_gt_20"] / max(1, n_done)
print(f"\n[GENUINE] successes ending after >10 steps: {_gen}/{_ge['n']} "
      f"=> genuine success rate {res['genuine_success_pct']:.2f}% of {n_done} episodes")
print(f"\n[EVAL_RESULT] {args.reset_type}: held-at-end {res['success_held_at_end_pct']:.1f}% | "
      f"ever-during {res['success_ever_during_pct']:.1f}% | n={n_done} | "
      f"|a| arm={res['mean_abs_action_arm']:.3f} hand={res['mean_abs_action_hand']:.3f}")
_p = res["goal_progress_m"]
print(f"[EVAL_RESULT] goal progress: median {_p['median']*1000:+.1f} mm "
      f"(p10 {_p['p10']*1000:+.1f}, p90 {_p['p90']*1000:+.1f}) | "
      f"episodes that got CLOSER by >5 mm: {_p['frac_closer']*100:.1f}%")
_d = res["object_dz_m"]
print(f"[EVAL_RESULT] object height change: median {_d['median']*1000:+.1f} mm "
      f"(p10 {_d['p10']*1000:+.1f}, p90 {_d['p90']*1000:+.1f})")
_e = res["position_error_m"]
print(f"[EVAL_RESULT] final error tail: min {_e['min']*1000:.1f} mm, p10 {_e['p10']*1000:.1f} mm | "
      f"within 20 mm: {_e['frac_within_20mm']*100:.1f}% | within 50 mm: {_e['frac_within_50mm']*100:.1f}%")
_z = res["final_object_z_m"]
print(f"[EVAL_RESULT] final cube height: median {_z['median']*1000:.1f} mm "
      f"(p10 {_z['p10']*1000:.1f}, p90 {_z['p90']*1000:.1f}) | "
      f"on the table at the end: {_z['frac_on_table']*100:.1f}% of {_z['n']} episodes")
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=2))
simulation_app.close()
