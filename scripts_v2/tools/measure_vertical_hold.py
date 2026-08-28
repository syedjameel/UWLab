# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""VERIFICATION GATE for bead UWLab-hlk7 (epic UWLab-nnlv): can a given DexLift table-leg Reorient
checkpoint, when COMMANDED a vertical (tip-down) goal, actually HOLD the leg vertical?

THE QUESTION THIS ANSWERS, in one number plus a distribution: among the states where the checkpoint
is genuinely HOLDING the leg (fingertip contact force, not in-palm displacement -- see below), how
close to vertical is it. Everything else this script prints is context for that one number.

LINEAGE. Modelled on ``measure_contact_grasp_closedloop.py`` (the FINGERTIP-CONTACT-FORCE grading
idiom: ``_sensor_force_magnitudes`` read directly off dexlift's own object-centric contact sensor,
THUMB_TIP_NAMES/OTHER_TIP_NAMES split, 0.2 N threshold -- reused verbatim, not re-picked) and
``replay_reset_states_policy.py`` (the AppLauncher/rl_games boot boilerplate, and the "verify the
plant off the CONSTRUCTED articulation, never the env var" idiom -- also reused verbatim, tightened
here from a warn-and-continue check into a hard REFUSAL). Both live in this directory.

WHY CONTACT FORCE AND NOT held_check.py's full ``held_with_probe``. ``held_with_probe`` ANDs four
gates, one of which is a probe that jogs the arm and checks whether the object's DISPLACEMENT
tracks the gripper's own displacement within an absolute-ish tolerance. That gate exists to catch an
object resting near an idle hand with no real grip -- a real concern for a stationary lift-and-hold
task. It is the WRONG tool here: measured on this exact checkpoint family, a 20 mm in-palm
displacement tolerance disagreed with contact-force grading 0.96 vs 0.54 at n=298, because it scores
the policy's own legitimate IN-HAND REPOSITIONING (rolling the leg to vertical is exactly that) as a
dropped grasp. This script grades holding by fingertip contact force ALONE: a thumb-side tip
(rl_dg_1 or rl_dg_5) AND at least one other tip (rl_dg_2/3/4) loaded above 0.2 N. It does not import
``held_with_probe`` or wire anything into ``terminations.success``, and injects no probe action.

THE MEASUREMENT LOOP READS STATE *BEFORE* CALLING ``env.step()``, NOT AFTER, and this is the one
design choice that differs from both model scripts and is worth stating plainly. IsaacLab's
``ManagerBasedRLEnv`` auto-resets a done env INSIDE the same ``env.step()`` call
(``episode_length_buf`` zeroed, scene tensors overwritten, before ``step()`` returns) -- so reading
``env.scene`` tensors AFTER ``step()`` returns is contaminated for any env whose ``done`` just fired
THIS step: it describes the brand-new episode that env was just teleported into, not the episode
that ended. Both model scripts sidestep this only by keeping the rollout window (2 s) far shorter
than the episode length (4 s), so it never happens. This script deliberately rolls for the WHOLE
episode (``--episode_length_s 8.0``, i.e. every surviving env hits ``time_out`` together at the same
final step) plus, optionally, several consecutive episodes per env (``--episodes``), so that
shortcut is not available: a post-step read at the terminal step would be contaminated for EVERY
env, not a rare few. The fix: read tilt and the contact-force held-gate ONCE PER ITERATION, BEFORE
calling ``env.step()`` -- i.e. the state as of the end of the PREVIOUS control tick (or the initial
``env.reset()`` for the very first iteration), which by construction cannot have been touched by a
reset that has not happened yet. This makes the per-step trace correct across an arbitrary number of
episode boundaries with no ``alive``/``died_at`` bookkeeping needed for the tilt/held numbers
themselves (unlike the model scripts, which need that bookkeeping precisely because they read
post-step). The lag this costs is at most one control tick (~16 ms at 60 Hz decimation=2), which is
irrelevant to a tilt-from-vertical or a fingertip-force decision. Termination CAUSE (time_out vs the
one other active termination, ``object_out_of_bound``) is still read immediately after ``step()``
returns, exactly as both model scripts do, because that read is inherently about what happened
during the step just taken and is used only for the informational episode-outcome counts below, not
for any tilt or held-gate number.

FIVE TILT SERIES, each reported as its own distribution:
  * BEST-ANY-STEP    -- per env, the minimum tilt-from-vertical seen at any point in the whole
                         rollout (which may span several episodes -- see --episodes), regardless of
                         whether the leg was held at that moment. "Can it ever get there at all."
                         Informational.
  * TERMINAL          -- per env, the tilt at the LAST pre-step reading of the whole rollout. A
                         terminal-only number has repeatedly under-reported what this project's
                         policies actually reach mid-episode (see the module docstring precedent in
                         omnireset-success-metric-terminal-only); reported for completeness, never
                         alone. Informational.
  * HELD (pooled)     -- every (env, step) pair across the whole rollout where the contact-force
                         held-gate is True, pooled into one distribution. INFORMATIONAL ONLY, NOT
                         THE GATE: consecutive steps of the same physical hold are highly
                         autocorrelated, so this n is not n independent draws, and its median
                         describes how long a good grasp LASTED at least as much as how many
                         independent attempts reached vertical (an env held for 200 steps
                         contributes 200 rows). Kept, printed, and compared against the per-env
                         median below -- a disagreement between the two is itself reported as a
                         finding, not hidden.
  * HELD (per-env best) -- per env that was ever held, the single best (minimum) tilt reached while
                         held. INFORMATIONAL ONLY, and deliberately NOT the gate either: it has the
                         SAME autocorrelation bias as the pooled series wearing a different hat (an
                         env held longer gets more draws, so it is more likely to have caught one
                         lucky low-tilt instant) -- printed purely as a supplementary cross-check.
  * HELD (per-env median) -- for each env that was EVER held, the MEDIAN tilt over THAT ENV's OWN
                         held steps, reducing its whole held history to one number; envs never held
                         contribute no sample. THIS IS THE GATE METRIC: one number per independent
                         rollout (per env), immune to both biases above.

THE GATE ALSO REQUIRES ENOUGH ENVS TO HAVE HELD AT ALL: a low per-env median over a handful of envs
is a fail dressed as a pass, so PASS additionally requires at least one quarter of --num_envs to have
been held at least once (n_envs_ever_held/num_envs is always printed beside the gate number, and the
GATE line spells out explicitly when this is what failed it).

BASELINES (measured 2026-08-26/27, printed again in this script's own output so a reader cannot
forget them while reading a result -- see the epic notes / goal_mixture.py's module docstring):
    shipped C3 bank (n=1800):                     min tilt 43 deg, 0/1800 under 25 deg
    ep_3600 + vertical goal, POSE_TILT clamp OPEN: min tilt 20.8 deg, 1/60 under 25 deg
    + goal pos_z constrained 0.06-0.15 m:          min tilt 14.5 deg, 2/60 under 25 deg
Those baselines count INDEPENDENT states/episodes, so the like-for-like number from this run is the
FRACTION OF ENVS that were EVER held with tilt under 25/15/10 deg -- reported beside the pooled
per-(env,step) counts, which are not like-for-like with the baselines' n.

GATE: PASS iff (a) the median of the PER-ENV HELD-median tilt values <= --median_tilt_gate_deg
(default 10 deg) AND (b) at least one quarter of --num_envs were ever held at all -- both printed
explicitly, so a good median over 3 of 64 envs cannot be mistaken for a pass. Meant to clearly beat
the 14.5 deg minimum / ~3 percent yield above, not merely edge past it.

MANDATORY ENVIRONMENT. This script SETS these itself (does not trust the caller's shell), because a
silently-unset one of them has repeatedly produced a plausible WRONG number rather than an error in
this project:
    DEXLIFT_REF_RESET=1 DEXLIFT_REF_ACTUATORS=1 DEXLIFT_REF_HAND_ACT=1 DEXLIFT_REF_ARM_ACT=0
    DEXLIFT_POSE_TILT=0.3
    DEXLIFT_GOAL_VERTICAL_PROB=1.0 (see --goal_vertical_prob; 1.0 is legitimate for a pure
        measurement -- goal_mixture.py only WARNS at 1.0, the training LAUNCHER is what refuses it)
    --episode_length_s 8.0 (registered default is 4.0; that alone was worth 3.6x acceptance upstream)
UWLAB_TMP_ROOT / TMPDIR default to $HOME/tmp if unset (DL_H100: /tmp/uwlab and /tmp/isaaclab are
owned by another uid, and TMPDIR is NOT covered by UWLAB_TMP_ROOT -- IsaacLab's logger calls
tempfile.gettempdir() directly).

PLANT VERIFICATION IS A HARD REFUSAL HERE, not a print-and-continue. Read off the CONSTRUCTED
articulation (``robot.data.joint_effort_limits`` / ``joint_velocity_limits`` on the hand joints,
exactly as ``replay_reset_states_policy.py`` reads them -- never the env var, never the cfg object,
which is one layer removed from what the sim actually built) after ``gym.make``, before the
checkpoint is even loaded: if the hand does not read effort 30.0 N*m / velocity 10000.0 rad/s, this
script prints ``VERTICAL_HOLD_REFUSED`` and exits nonzero without producing a tilt number of any
kind, rather than a wrong one.

Run (one Isaac process; never via uwlab.sh; never two Isaac processes on one GPU -- silently invalid
physics at rc=0):
    PYTHONPATH="$PWD/source/uwlab:$PWD/source/uwlab_tasks:$PWD/source/uwlab_assets:$PWD/source/uwlab_rl" \\
        timeout -s KILL 900 <python> -u scripts_v2/tools/measure_vertical_hold.py \\
        --checkpoint <path>.pth --num_envs 64 --episodes 1 --episode_length_s 8.0 \\
        --device cuda:0 --out vertical_hold_result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# ==================================== BASELINES, PRINTED VERBATIM ====================================
# (measured 2026-08-26/27; see this file's module docstring for provenance). Kept as one literal
# dict, not scattered f-strings, so the numbers in the docstring and the numbers actually printed at
# runtime cannot drift apart.
BASELINES = {
    "shipped_C3_bank_n1800": {"n": 1800, "min_tilt_deg": 43.0, "frac_under_25deg": 0.0},
    "ep3600_vertical_goal_tilt_clamp_open_n60": {"n": 60, "min_tilt_deg": 20.8, "frac_under_25deg": 1 / 60},
    "ep3600_vertical_goal_posz_0p06_0p15_n60": {"n": 60, "min_tilt_deg": 14.5, "frac_under_25deg": 2 / 60},
}

# ==================================== ARGPARSE ====================================
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint to grade.")
parser.add_argument(
    "--agent_yaml", type=str, default=None,
    help="Path to the checkpoint's params/agent.yaml. Defaults to <checkpoint's params dir>/agent.yaml.",
)
parser.add_argument(
    "--task", type=str, default="DexLift-UR5eDelto-RelJointPos-TableLeg-Reorient-Play-v0",
    help="The PLAY variant of the table-leg Reorient task (debug_vis on, single-env-friendly resets).",
)
parser.add_argument("--num_envs", type=int, default=64)
_episode_group = parser.add_mutually_exclusive_group()
_episode_group.add_argument(
    "--episodes", type=int, default=None,
    help="Roll out this many consecutive episodes per env (steps = episodes * env.max_episode_length, "
         "read off the CONSTRUCTED env, not computed by hand). Default 1 if neither this nor --steps given.",
)
_episode_group.add_argument(
    "--steps", type=int, default=None,
    help="Roll out exactly this many control steps per env instead of a whole-number-of-episodes count.",
)
parser.add_argument(
    "--episode_length_s", type=float, default=8.0,
    help="Override env_cfg.episode_length_s (this run's env_cfg copy only). Default 8.0, NOT the "
         "registered 4.0 -- measured to be worth 3.6x acceptance on its own. Pass the registered "
         "value explicitly if you deliberately want to reproduce the shorter-episode regime.",
)
parser.add_argument(
    "--pose_tilt", type=float, default=0.3,
    help="DEXLIFT_POSE_TILT (rad). Default 0.3 matches the certified lineage this checkpoint was "
         "trained/finetuned under; changing it measures a different (and here, unvalidated) task.",
)
parser.add_argument(
    "--goal_vertical_prob", type=float, default=1.0,
    help="DEXLIFT_GOAL_VERTICAL_PROB. 1.0 (default) forces EVERY resampled goal vertical, which is "
         "the correct setting for a pure measurement of 'can it hold vertical when told to' -- "
         "goal_mixture.py only warns at 1.0 and does not refuse; the training LAUNCHER is what "
         "refuses it (a 100%% finetune mixture has destroyed this policy's parent skill before, "
         "which is a training-time concern, not a measurement-time one).",
)
parser.add_argument(
    "--goal_vertical_tilt", type=float, default=None,
    help="DEXLIFT_GOAL_VERTICAL_TILT (rad half-width). Default None leaves the module's own default "
         "(0.35 rad = 20 deg) in place -- i.e. whatever the finetune was actually trained under.",
)
parser.add_argument(
    "--goal_vertical_z", type=str, default=None,
    help="DEXLIFT_GOAL_VERTICAL_Z as 'lo,hi' metres (root frame). Default None leaves the module's "
         "own default ('0.13,0.27') in place. The 14.5 deg / 2-of-60 baseline above used a TIGHTER, "
         "ad hoc '0.06,0.15' on a DIFFERENT (pre-finetune) checkpoint -- pass it explicitly here only "
         "if you intend to reproduce that specific comparison, not by default.",
)
parser.add_argument(
    "--force_threshold", type=float, default=0.2,
    help="N, held_with_probe's / measure_contact_grasp_closedloop.py's own default fingertip-force "
         "gate. Reused, not re-picked.",
)
parser.add_argument(
    "--median_tilt_gate_deg", type=float, default=10.0,
    help="GATE threshold: PASS iff the MEDIAN OF THE PER-ENV HELD-median tilt-from-vertical values "
         "is <= this many degrees (one number per env that was ever held, not one per (env,step) -- "
         "see the module docstring's FOUR TILT SERIES section for why). Must be well under the "
         "14.5 deg / ~3%% baseline above to count as clearly beating it, not just edging past it -- "
         "the default (10 deg) is chosen for that margin.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Path to write the JSON result to.")
AppLauncher.add_app_launcher_args(parser)  # provides --device, among others
args_cli = parser.parse_args()

# ==================================== CHECKPOINT IDENTITY ====================================
# Computed off the BYTES, before anything else touches the file. Two checkpoints in this project
# differ only by an epoch number in the filename and rank OPPOSITE ways on training reward vs
# certified score -- a filename is not an identity.
_ckpt_path = Path(args_cli.checkpoint)
if not _ckpt_path.is_file():
    raise SystemExit(f"[vhold] REFUSING: checkpoint not found: {_ckpt_path}")
_sha256 = hashlib.sha256()
with open(_ckpt_path, "rb") as _f:
    for _chunk in iter(lambda: _f.read(1 << 20), b""):
        _sha256.update(_chunk)
CKPT_SHA256 = _sha256.hexdigest()
print(f"[vhold] checkpoint = {_ckpt_path}", flush=True)
print(f"[vhold] checkpoint sha256 = {CKPT_SHA256}", flush=True)

# ==================================== MANDATORY ENVIRONMENT ====================================
# Set here, unconditionally, BEFORE parse_env_cfg -- never trusted from the caller's shell. See the
# module docstring's "MANDATORY ENVIRONMENT" section for why each one is here.
os.environ["DEXLIFT_REF_RESET"] = "1"
os.environ["DEXLIFT_REF_ACTUATORS"] = "1"
os.environ["DEXLIFT_REF_ARM_ACT"] = "0"
os.environ["DEXLIFT_REF_HAND_ACT"] = "1"
os.environ["DEXLIFT_POSE_TILT"] = str(args_cli.pose_tilt)
os.environ["DEXLIFT_GOAL_VERTICAL_PROB"] = str(args_cli.goal_vertical_prob)
if args_cli.goal_vertical_tilt is not None:
    os.environ["DEXLIFT_GOAL_VERTICAL_TILT"] = str(args_cli.goal_vertical_tilt)
if args_cli.goal_vertical_z is not None:
    os.environ["DEXLIFT_GOAL_VERTICAL_Z"] = args_cli.goal_vertical_z
os.environ.setdefault("DEXLIFT_SPAWN_CLEARANCE", "1")
# DL_H100: /tmp/uwlab and /tmp/isaaclab are owned by another uid; TMPDIR is NOT covered by
# UWLAB_TMP_ROOT (IsaacLab's logger calls tempfile.gettempdir() directly). Harmless elsewhere.
_tmp_root = os.environ.get("UWLAB_TMP_ROOT", str(Path.home() / "tmp"))
os.environ.setdefault("UWLAB_TMP_ROOT", _tmp_root)
os.environ.setdefault("TMPDIR", _tmp_root)
Path(os.environ["UWLAB_TMP_ROOT"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

if args_cli.episode_length_s != 8.0:
    print(
        f"[vhold] WARNING: --episode_length_s={args_cli.episode_length_s}, NOT the mandated 8.0. "
        f"Any result from this run describes a different (and here, unvalidated) measurement regime.",
        flush=True,
    )

import torch  # noqa: E402
import yaml  # noqa: E402

# ==================================== QUAT_APPLY SELF-TEST ====================================
# Hand-rolled, (w, x, y, z) convention, exactly like analyze_grasp_orientation_distribution.py's own
# -- deliberately NOT isaaclab.utils.math's, to sidestep the WXYZ/XYZW convention trap that has
# silently corrupted at least one other term in this project. Self-tested against the exact worked
# example this script's own tilt definition was specified with, before anything expensive runs.


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (..., 3) by unit quaternion q (..., 4), (w, x, y, z) convention."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


_TIP_AXIS_LOCAL = torch.tensor([-1.0, 0.0, 0.0])  # leg's local -X = tip direction
_WORLD_DOWN = torch.tensor([0.0, 0.0, -1.0])
_ry_neg90 = torch.tensor([0.70710678, 0.0, -0.70710678, 0.0])  # (w, x, y, z)
_check = quat_apply(_ry_neg90, _TIP_AXIS_LOCAL)
assert torch.allclose(_check, _WORLD_DOWN, atol=1e-4), (
    f"[vhold] REFUSING: quat_apply self-test failed -- Ry(-90) applied to the tip axis gave "
    f"{_check.tolist()}, expected {_WORLD_DOWN.tolist()}. The tilt-from-vertical definition below "
    f"would be silently wrong."
)
print("[vhold] quat_apply self-test OK: Ry(-90) tip axis -> world (0, 0, -1) as specified", flush=True)


def leg_tilt_deg(obj_quat_w: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the leg's world tip axis (local -X) and world down (0, 0, -1)."""
    tip_axis_local = _TIP_AXIS_LOCAL.to(obj_quat_w.device).expand(obj_quat_w.shape[0], 3)
    tip_world = quat_apply(obj_quat_w, tip_axis_local)
    tip_world = tip_world / torch.linalg.norm(tip_world, dim=-1, keepdim=True).clamp_min(1e-8)
    down = _WORLD_DOWN.to(obj_quat_w.device)
    cos_ang = (tip_world * down).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_ang))


# ==================================== BOOT ISAAC ====================================
print(f"[vhold] num_envs={args_cli.num_envs} task={args_cli.task}", flush=True)
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

# REUSED VERBATIM from measure_contact_grasp_closedloop.py -- same function held_check.py calls,
# same thumb/tip name split, same 0.2 N default. See this file's module docstring for why this (and
# not held_with_probe's displacement probe) is the right primitive to grade holding with here.
from uwlab_tasks.manager_based.manipulation.dexlift.mdp.rewards import _sensor_force_magnitudes  # noqa: E402

THUMB_TIP_NAMES = ("rl_dg_1_tip", "rl_dg_5_tip")
OTHER_TIP_NAMES = ("rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip")

cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
cfg.seed = args_cli.seed
cfg.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024  # same budget both model scripts use
_original_episode_length_s = cfg.episode_length_s
cfg.episode_length_s = args_cli.episode_length_s
print(
    f"[vhold] episode_length_s OVERRIDE (this env_cfg copy only): "
    f"{_original_episode_length_s} -> {cfg.episode_length_s}",
    flush=True,
)
# NOTE, not fixed here: the Reorient _PLAY class's own commands.object_pose.resampling_time_range is
# (2.0, 3.0), well UNDER an 8 s episode, and this script does not re-derive it (unlike
# generate_reset_states_policy.py's handling of the DIFFERENT GoalBelowSpawnPoseCommand). At
# goal_vertical_prob=1.0 every resample -- including any mid-episode one this causes -- still draws
# from the SAME vertical band, so the commanded goal stays "vertical" throughout; only its specific
# height/orientation within that band can shift mid-episode. Left as-is because it does not change
# what this script measures (tilt-from-vertical of the leg, not tracking of one fixed target pose);
# flagged here so it is a documented choice, not a missed one.

env = gym.make(args_cli.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
obj = env.scene["object"]  # dexlift's own name for the leg

# ==================================== HARD PLANT REFUSAL ====================================
# Read off the CONSTRUCTED articulation, never the env var and never the cfg object -- one step
# further than replay_reset_states_policy.py's own check, which only compares reference-vs-
# identified; this one REFUSES outright if the built hand is not exactly the reference plant this
# whole measurement assumes.
_hand_joint_ids, _ = robot.find_joints([r"rj_dg_[1-5]_[1-4]"], preserve_order=False)
_hand_eff = robot.data.joint_effort_limits[0, _hand_joint_ids]
_hand_vel = robot.data.joint_velocity_limits[0, _hand_joint_ids]
_eff_lo, _eff_hi = float(_hand_eff.min()), float(_hand_eff.max())
_vel_lo, _vel_hi = float(_hand_vel.min()), float(_hand_vel.max())
print(f"[vhold] hand AS BUILT: effort_limit_sim {_eff_lo:.3f}-{_eff_hi:.3f} N*m, "
      f"velocity_limit_sim {_vel_lo:.1f}-{_vel_hi:.1f} rad/s  (expect 30.0 / 10000.0)", flush=True)


def _refuse(reason: str) -> None:
    print(f"[vhold] REFUSING to produce a number: {reason}", flush=True)
    Path(args_cli.out).write_text(json.dumps(
        {"status": "REFUSED", "reason": reason, "checkpoint": str(_ckpt_path), "checkpoint_sha256": CKPT_SHA256},
        indent=2,
    ))
    print("VERTICAL_HOLD_REFUSED", flush=True)
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(1)


if not (math.isclose(_eff_lo, 30.0, abs_tol=1e-3) and math.isclose(_eff_hi, 30.0, abs_tol=1e-3)):
    _refuse(f"hand effort_limit_sim as BUILT is {_eff_lo:.3f}-{_eff_hi:.3f} N*m, not 30.0 N*m. "
             f"DEXLIFT_REF_HAND_ACT did not silently take effect the way this measurement assumes.")
if not (math.isclose(_vel_lo, 10000.0, abs_tol=1.0) and math.isclose(_vel_hi, 10000.0, abs_tol=1.0)):
    _refuse(f"hand velocity_limit_sim as BUILT is {_vel_lo:.1f}-{_vel_hi:.1f} rad/s, not 10000.0 rad/s. "
             f"DEXLIFT_REF_HAND_ACT did not silently take effect the way this measurement assumes.")
print("[vhold] PLANT VERIFICATION OK (reference hand: 30.0 N*m / 10000.0 rad/s)", flush=True)

# ==================================== LOAD THE POLICY ====================================
agent_yaml = args_cli.agent_yaml or os.path.join(Path(args_cli.checkpoint).parent.parent, "params", "agent.yaml")
agent_cfg = yaml.safe_load(open(agent_yaml))
clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
wrapped = RlGamesVecEnvWrapper(env, args_cli.device, clip_obs, clip_act,
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kw: RlGamesGpuEnv(name, num_actors, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
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
print("[vhold] POLICY_LOADED", flush=True)


def unwrap(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o["obs"]
    return o


obs = unwrap(wrapped.reset())
dev = env.device

# ==================================== STEP BUDGET ====================================
max_ep_len = int(env.max_episode_length)  # read off the CONSTRUCTED env, not computed by hand
print(f"[vhold] env.max_episode_length = {max_ep_len} steps  (step_dt={env.step_dt:.5f} s, "
      f"episode_length_s={cfg.episode_length_s})", flush=True)
if args_cli.steps is not None:
    n_steps = args_cli.steps
elif args_cli.episodes is not None:
    n_steps = args_cli.episodes * max_ep_len
else:
    n_steps = max_ep_len  # default: exactly one episode
print(f"[vhold] rolling out {n_steps} control steps ({n_steps / max_ep_len:.2f} episodes/env)", flush=True)

# ==================================== ROLLOUT ====================================
# See the module docstring's "THE MEASUREMENT LOOP READS STATE BEFORE CALLING env.step()" section
# for why every tensor read below happens BEFORE the step() call it precedes, not after.
N = env.num_envs
best_tilt_deg = torch.full((N,), float("inf"), device=dev)
terminal_tilt_deg = torch.zeros(N, device=dev)
held_ever = torch.zeros(N, dtype=torch.bool, device=dev)
held_tilt_pooled: list[float] = []
per_env_best_held_tilt = torch.full((N,), float("inf"), device=dev)
# PER-ENV held tilts, kept as one list per env, because THE GATE IS A PER-ENV STATISTIC and the
# pooled one cannot be. Pooled (env, step) samples are AUTOCORRELATED: an env that holds for 200
# steps contributes 200 rows, so a pooled median describes how long a good grasp LASTED rather than
# how many independent attempts reached vertical. Reducing each env to one number first removes
# that weighting. The reduction is each env's MEDIAN over its own held steps, NOT its best -- a
# per-env best is an extremum over a variable number of draws, so an env that held longer wins by
# having had more chances, which is the same bias wearing a different hat.
per_env_held_tilts: list[list[float]] = [[] for _ in range(N)]
# Yield-like counters, in the units the baselines are quoted in: the baselines (0/1800, 1/60, 2/60
# under 25 deg) count INDEPENDENT STATES, so the like-for-like comparison is a fraction of ENVS
# that ever reached held-and-under-threshold -- never a fraction of pooled steps.
held_under_thresh_ever = {t: torch.zeros(N, dtype=torch.bool, device=dev) for t in (25.0, 15.0, 10.0)}
n_time_out_events = 0
n_object_out_of_bound_events = 0

for step in range(n_steps):
    with torch.inference_mode():
        # -- CLEAN pre-step read: reflects the end of the PREVIOUS control tick (or the initial
        # reset for step 0). Cannot be contaminated by a reset that has not happened yet.
        tilt = leg_tilt_deg(obj.data.root_quat_w)
        thumb_force = _sensor_force_magnitudes(env, THUMB_TIP_NAMES)
        other_force = _sensor_force_magnitudes(env, OTHER_TIP_NAMES)
        held = (thumb_force > args_cli.force_threshold).any(dim=-1) & (other_force > args_cli.force_threshold).any(dim=-1)

        best_tilt_deg = torch.minimum(best_tilt_deg, tilt)
        terminal_tilt_deg = tilt  # overwritten every iteration; holds the LAST pre-step value after the loop
        held_ever = held_ever | held
        per_env_best_held_tilt = torch.where(held, torch.minimum(per_env_best_held_tilt, tilt), per_env_best_held_tilt)
        for thr, buf in held_under_thresh_ever.items():
            buf |= held & (tilt <= thr)
        if bool(held.any()):
            held_tilt_pooled.extend(tilt[held].cpu().tolist())
            held_idx = held.nonzero(as_tuple=False).flatten().cpu().tolist()
            held_vals = tilt[held].cpu().tolist()
            for env_i, val in zip(held_idx, held_vals):
                per_env_held_tilts[env_i].append(val)

        # -- act and step. Termination CAUSE is read immediately after, for the informational
        # episode-outcome counts only (never for a tilt/held number -- see module docstring).
        act = player.get_action(obs, is_deterministic=True)
        ret = wrapped.step(act)
        obs = unwrap(ret[0])
        done = ret[2].to(dev).bool()
        if bool(done.any()):
            tm = env.termination_manager
            if "time_out" in tm.active_terms:
                n_time_out_events += int((tm.get_term("time_out") & done).sum())
            if "object_out_of_bound" in tm.active_terms:
                n_object_out_of_bound_events += int((tm.get_term("object_out_of_bound") & done).sum())

print(f"\n[vhold] episode-end events over the rollout: time_out={n_time_out_events}  "
      f"object_out_of_bound={n_object_out_of_bound_events}", flush=True)

# ==================================== STATS ====================================
_THRESHOLDS = (25.0, 15.0, 10.0)


def dist_stats(t: torch.Tensor) -> dict:
    t = t[torch.isfinite(t)]
    n = int(t.numel())
    if n == 0:
        return {"n": 0}
    qs = torch.quantile(t, torch.tensor([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], device=t.device))
    out = {
        "n": n,
        "min": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]),
        "median": float(qs[3]), "p75": float(qs[4]), "max": float(qs[5]),
    }
    for thr in _THRESHOLDS:
        out[f"frac_under_{int(thr)}deg"] = float((t < thr).float().mean())
    return out


def print_dist(label: str, d: dict) -> None:
    if d["n"] == 0:
        print(f"[vhold] {label}: n=0 (no samples)", flush=True)
        return
    print(f"[vhold] {label}: n={d['n']}  min={d['min']:.2f}  p10={d['p10']:.2f}  p25={d['p25']:.2f}  "
          f"median={d['median']:.2f}  p75={d['p75']:.2f}  max={d['max']:.2f}  "
          f"frac<25={d['frac_under_25deg']:.3f}  frac<15={d['frac_under_15deg']:.3f}  "
          f"frac<10={d['frac_under_10deg']:.3f}", flush=True)


best_stats = dist_stats(best_tilt_deg.cpu())
terminal_stats = dist_stats(terminal_tilt_deg.cpu())
held_pooled_stats = dist_stats(torch.tensor(held_tilt_pooled) if held_tilt_pooled else torch.zeros(0))
held_per_env_stats = dist_stats(per_env_best_held_tilt.cpu())

# THE GATE SERIES: one number per env that was ever held -- that env's MEDIAN tilt over its own held
# steps. Envs never held contribute NO tilt sample and are counted separately below; folding them in
# as a large tilt would invent a measurement, and dropping them silently would let a 5 deg median
# over 3 of 64 envs read as a pass.
_per_env_medians = [
    float(torch.tensor(vals).median()) for vals in per_env_held_tilts if len(vals) > 0
]
n_envs_held = len(_per_env_medians)
held_per_env_median_stats = dist_stats(
    torch.tensor(_per_env_medians) if _per_env_medians else torch.zeros(0)
)
frac_env_under = {
    thr: float(buf.float().mean()) for thr, buf in held_under_thresh_ever.items()
}

print("\n[vhold] ==================== BASELINES (context, not measured by this run) ====================", flush=True)
for name, b in BASELINES.items():
    print(f"[vhold]   {name}: n={b['n']}  min_tilt_deg={b['min_tilt_deg']}  "
          f"frac_under_25deg={b['frac_under_25deg']:.4f}", flush=True)

print("\n[vhold] ==================== TILT DISTRIBUTIONS (this run) ====================", flush=True)
print_dist("BEST-ANY-STEP    (per env, min tilt anywhere in the rollout, held or not)", best_stats)
print_dist("TERMINAL         (per env, tilt at the last pre-step reading of the rollout)", terminal_stats)
print_dist("HELD (pooled)    (every (env,step) held -- INFORMATIONAL, autocorrelated, NOT the gate)", held_pooled_stats)
print_dist("HELD (per-env best) (per env ever held, best tilt while held -- informational)", held_per_env_stats)
print_dist("HELD (per-env MEDIAN) (one number per env ever held -- THIS IS THE GATE METRIC)", held_per_env_median_stats)
print(f"[vhold] held_ever: {int(held_ever.sum())}/{N} envs ({float(held_ever.float().mean()):.3f}) were "
      f"held by contact force at least once during the rollout", flush=True)
print(f"[vhold] YIELD, like-for-like with the baselines (which count INDEPENDENT states, not steps): "
      f"fraction of ENVS ever held with tilt <=25deg {frac_env_under[25.0]:.3f} "
      f"({int(held_under_thresh_ever[25.0].sum())}/{N}), <=15deg {frac_env_under[15.0]:.3f} "
      f"({int(held_under_thresh_ever[15.0].sum())}/{N}), <=10deg {frac_env_under[10.0]:.3f} "
      f"({int(held_under_thresh_ever[10.0].sum())}/{N})", flush=True)

# -- A DISAGREEMENT BETWEEN THE TWO HELD SERIES IS ITSELF A FINDING, so it is printed rather than
# left for a reader to notice. They measure different things (see the module docstring): pooled is
# weighted by how long each hold LASTED, per-env is one vote per independent rollout.
if held_pooled_stats["n"] > 0 and held_per_env_median_stats["n"] > 0:
    _delta = abs(held_pooled_stats["median"] - held_per_env_median_stats["median"])
    if _delta > 2.0:
        print(f"[vhold] NOTE: pooled median {held_pooled_stats['median']:.2f} deg and per-env median "
              f"{held_per_env_median_stats['median']:.2f} deg DISAGREE by {_delta:.2f} deg. The gate "
              "uses the per-env number; the gap means hold DURATION correlates with tilt (long holds "
              "are systematically better or worse aligned than short ones), which is worth reading "
              "before acting on either number.", flush=True)

# ==================================== GATE ====================================
# KEYED OFF THE PER-ENV SERIES, never the pooled one -- see the module docstring for why a pooled
# (env, step) median cannot decide this.
if held_per_env_median_stats["n"] == 0:
    gate_pass = False
    gate_value = float("nan")
    gate_reason = f"no held states observed in ANY of {N} envs -- automatic FAIL"
else:
    gate_value = held_per_env_median_stats["median"]
    _enough_envs = n_envs_held >= max(1, N // 4)
    gate_pass = (gate_value <= args_cli.median_tilt_gate_deg) and _enough_envs
    gate_reason = (
        f"per-env HELD median tilt {gate_value:.2f} deg "
        f"{'<=' if gate_value <= args_cli.median_tilt_gate_deg else '>'} gate threshold "
        f"{args_cli.median_tilt_gate_deg:.2f} deg, over {n_envs_held}/{N} envs that were ever held"
        + ("" if _enough_envs else
           f" -- REFUSED: fewer than a quarter of envs ({N // 4}) ever held, so this median describes"
           " too few independent rollouts to be a gate whatever its value")
    )

print(f"\n[vhold] GATE={'PASS' if gate_pass else 'FAIL'}  ({gate_reason})", flush=True)
print(f"[vhold] for comparison: best-baseline-so-far min_tilt_deg=14.5 at ~3 percent yield "
      f"(ep3600_vertical_goal_posz_0p06_0p15_n60) -- this run's HELD median must clearly beat that, "
      f"not merely edge past the {args_cli.median_tilt_gate_deg:.1f} deg gate number.", flush=True)

# ==================================== JSON OUTPUT ====================================
result = {
    "status": "OK",
    "checkpoint": str(_ckpt_path),
    "checkpoint_sha256": CKPT_SHA256,
    "agent_yaml": agent_yaml,
    "task": args_cli.task,
    "num_envs": N,
    "seed": args_cli.seed,
    "episode_length_s": cfg.episode_length_s,
    "max_episode_length_steps": max_ep_len,
    "n_steps": n_steps,
    "force_threshold_N": args_cli.force_threshold,
    "env": {
        "DEXLIFT_REF_RESET": os.environ["DEXLIFT_REF_RESET"],
        "DEXLIFT_REF_ACTUATORS": os.environ["DEXLIFT_REF_ACTUATORS"],
        "DEXLIFT_REF_HAND_ACT": os.environ["DEXLIFT_REF_HAND_ACT"],
        "DEXLIFT_REF_ARM_ACT": os.environ["DEXLIFT_REF_ARM_ACT"],
        "DEXLIFT_POSE_TILT": os.environ["DEXLIFT_POSE_TILT"],
        "DEXLIFT_GOAL_VERTICAL_PROB": os.environ["DEXLIFT_GOAL_VERTICAL_PROB"],
        "DEXLIFT_GOAL_VERTICAL_TILT": os.environ.get("DEXLIFT_GOAL_VERTICAL_TILT", "<module default 0.35>"),
        "DEXLIFT_GOAL_VERTICAL_Z": os.environ.get("DEXLIFT_GOAL_VERTICAL_Z", "<module default 0.13,0.27>"),
    },
    "plant_as_built": {
        "hand_effort_limit_sim_Nm": [_eff_lo, _eff_hi],
        "hand_velocity_limit_sim_rad_s": [_vel_lo, _vel_hi],
    },
    "episode_end_events": {
        "time_out": n_time_out_events,
        "object_out_of_bound": n_object_out_of_bound_events,
    },
    "baselines": BASELINES,
    "tilt_deg": {
        "best_any_step": best_stats,
        "terminal": terminal_stats,
        "held_pooled_INFORMATIONAL_ONLY": held_pooled_stats,
        "held_per_env_best_INFORMATIONAL_ONLY": held_per_env_stats,
        "held_per_env_median_GATE_SERIES": held_per_env_median_stats,
    },
    "held_ever_fraction": float(held_ever.float().mean()),
    "n_envs_ever_held": n_envs_held,
    "yield_fraction_envs_ever_held_under_threshold": {
        f"{int(thr)}deg": frac_env_under[thr] for thr in (25.0, 15.0, 10.0)
    },
    "held_series_disagreement": (
        {
            "pooled_median_deg": held_pooled_stats["median"],
            "per_env_median_deg": held_per_env_median_stats["median"],
            "delta_deg": abs(held_pooled_stats["median"] - held_per_env_median_stats["median"]),
        }
        if held_pooled_stats["n"] > 0 and held_per_env_median_stats["n"] > 0
        else None
    ),
    "gate": {
        "metric": "median of the PER-ENV HELD-median tilt-from-vertical values (deg), one number per "
                  "env ever held -- NOT the pooled (env,step) median (see module docstring)",
        "value": gate_value,
        "threshold_deg": args_cli.median_tilt_gate_deg,
        "n_envs_ever_held": n_envs_held,
        "num_envs": N,
        "pass": gate_pass,
        "reason": gate_reason,
    },
}
Path(args_cli.out).write_text(json.dumps(result, indent=2))
print(f"\n[vhold] wrote {args_cli.out}", flush=True)

print(
    f"\nVERTICAL_HOLD_RESULT gate={'PASS' if gate_pass else 'FAIL'} "
    f"held_per_env_median_tilt_deg={gate_value:.3f} n_envs_ever_held={n_envs_held}/{N} "
    f"held_pooled_median_deg={held_pooled_stats.get('median', float('nan')):.3f} held_pooled_n={held_pooled_stats['n']} "
    f"held_ever_frac={float(held_ever.float().mean()):.3f} "
    f"frac_envs_under10deg={frac_env_under[10.0]:.3f} frac_envs_under15deg={frac_env_under[15.0]:.3f} "
    f"frac_envs_under25deg={frac_env_under[25.0]:.3f} "
    f"best_any_step_median_deg={best_stats.get('median', float('nan')):.3f} "
    f"terminal_median_deg={terminal_stats.get('median', float('nan')):.3f} "
    f"checkpoint_sha256={CKPT_SHA256}",
    flush=True,
)

import sys as _sys  # noqa: E402
_sys.stdout.flush()
_sys.stderr.flush()
os._exit(0 if gate_pass else 1)
