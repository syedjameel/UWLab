"""Record the FULL pose of every robot link through a policy rollout, for real mesh rendering.

WHY THIS EXISTS. The earlier collectors recorded three points (cube, two fingertips), which is enough
for a state-trace plot and nothing else. Matte asked to SEE the robot do the task, and a scatter plot
is not that. Isaac's RTX renderer draws empty sky on this box (six real bugs fixed, camera verifiably
correct, IOMMU + driver 610 against a stack wanting 580), so instead of rendering inside Isaac this
records what physics knows -- position and orientation of every link, every step -- and the robot is
rendered afterwards from its own USD meshes by an offscreen renderer that never touches the NVIDIA
path.

Physics has always worked here. Only the picture was missing.

Writes .npz (not JSON): 64 envs x 150 steps x ~40 bodies x 7 floats is large, and npz keeps it exact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

try:
    import uwlab_is6_bridge

    uwlab_is6_bridge.apply()
except ImportError:
    pass

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, "/home/mattephi/Work/Rubitek/UWLab-cubestack/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

TASK = "OmniReset-UR10eDelto-CubeStackTwoFingerGripCarry-RelCartesianOSC-State-v0"
N = int(os.environ.get("TRACE_ENVS", "64"))
STEPS = int(os.environ.get("TRACE_STEPS", "160"))
NAME = os.environ["TRACE_NAME"]
CK = os.environ["TRACE_CK"]
FAMILY = os.environ.get("TRACE_FAMILY", "ObjectPartiallyAssembledEEGrasped")
OUT = os.environ.get("TRACE_OUT", f"/home/mattephi/Work/Rubitek/cube-stack/results/poses_{NAME}.npz")
_t = lambda x: getattr(x, "torch", x)  # noqa: E731

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
ev = cfg.events.reset_from_reset_states
ev.params["reset_types"] = [FAMILY]
ev.params["probs"] = [1.0]

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

env_raw = gym.make(TASK, cfg=cfg)
u = env_raw.unwrapped
env = RslRlVecEnvWrapper(env_raw, clip_actions=getattr(agent_cfg, "clip_actions", None))

# TRACE_CK=none runs the NULL ACTION through this identical wrapper, reward path and contact sensor,
# so the effort-cap claim can be re-tested against MEASURED FORCE rather than the `in_hand` distance
# proxy it was originally built on. trace_collect_policy.py already had this branch; this file did
# not, and called runner.load("none") -> FileNotFoundError.
if CK.lower() in ("none", "null", ""):
    _z = torch.zeros((N, u.action_manager.total_action_dim), device=u.device)

    def policy(_obs):  # noqa: D401 -- same call signature as the inference policy
        return _z

    print("POSES NULL ACTION (no checkpoint); identical wrapper and contact sensor", flush=True)
else:
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CK)
    policy = runner.get_inference_policy(device=u.device)

ctx = u.reward_manager.get_term_cfg("progress_context").func
cmd = u.command_manager.get_term("task_command")
robot = u.scene["robot"]
obj = u.scene["insertive_object"]
goal_asset = u.scene["receptive_object"]
body_names = list(robot.body_names)
origins = _t(u.scene.env_origins)[:, :3]
print(f"POSES bodies={len(body_names)}: {body_names}", flush=True)

# Reset first: get_observations() does NOT start a fresh episode, and envs then hit their individual
# time-outs inside the recording window, which silently corrupts everything downstream.
_r = env.reset()
obs = _r[0] if isinstance(_r, tuple) else _r
if isinstance(obs, tuple):
    obs = obs[0]

body_pos, body_quat, cube_pos, cube_quat, goal_pos, goal_quat, succ, xyz = [], [], [], [], [], [], [], []
contact: list = []   # measured fingertip-cube contact force magnitudes, per env per tip
with torch.inference_mode():
    for step in range(STEPS):
        out = env.step(policy(obs))
        obs = out[0]
        if isinstance(obs, tuple):
            obs = obs[0]
        # LINK frame, not centre-of-mass. `body_pos_w` reports the COM, which is offset from the
        # link origin the USD meshes are authored around -- posing meshes by it produced a robot
        # whose wrist links sat 0.13-0.21 m apart from each other. `body_link_pos_w` is this repo's
        # own link-frame accessor (uwlab/assets/articulation/articulation_data.py:732) and is what
        # the repo's own reward terms use. Falls back only if the property is absent.
        _bp = getattr(robot.data, "body_link_pos_w", None)
        _bq = getattr(robot.data, "body_link_quat_w", None)
        if _bp is None or _bq is None:
            _bp, _bq = robot.data.body_pos_w, robot.data.body_quat_w
        bp = (_t(_bp) - origins[:, None, :]).cpu().numpy()
        bq = _t(_bq).cpu().numpy()          # wxyz
        body_pos.append(bp.astype(np.float32))
        body_quat.append(bq.astype(np.float32))
        cube_pos.append((_t(obj.data.root_pos_w) - origins).cpu().numpy().astype(np.float32))
        cube_quat.append(_t(obj.data.root_quat_w).cpu().numpy().astype(np.float32))
        goal_pos.append((_t(goal_asset.data.root_pos_w) - origins).cpu().numpy().astype(np.float32))
        goal_quat.append(_t(goal_asset.data.root_quat_w).cpu().numpy().astype(np.float32))
        succ.append(_t(ctx.success).cpu().numpy().astype(bool))
        xyz.append(_t(cmd.xyz_distance).cpu().numpy().astype(np.float32))
        # MEASURED CONTACT, not a distance proxy. `in_hand` is |d - d0| < 0.02 -- the CHANGE in
        # cube-to-fingertip distance -- so it reports a perfect grasp for a cube hanging motionless
        # 56 mm from wide-open fingers, which is exactly what it did for every rollout here. The
        # scene already carries `object_hand_s`, a ContactSensorCfg on the cube filtered to the five
        # fingertip prims; read it instead of inferring grasp from geometry.
        try:
            _s = u.scene["object_hand_s"]
            _fm = getattr(_s.data, "force_matrix_w", None)
            if _fm is None:
                _fm = getattr(_s.data, "net_forces_w", None)
            _f = _t(_fm).reshape(_fm.shape[0], -1, 3).norm(dim=-1)   # (envs, contacts)
            contact.append(_f.cpu().numpy().astype(np.float32))
        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never kill the rollout
            if step == 0:
                print(f"POSES contact sensor unavailable ({exc.__class__.__name__}: {exc})", flush=True)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT,
    body_pos=np.stack(body_pos), body_quat=np.stack(body_quat),
    cube_pos=np.stack(cube_pos), cube_quat=np.stack(cube_quat),
    goal_pos=np.stack(goal_pos), goal_quat=np.stack(goal_quat),
    success=np.stack(succ), xyz_distance=np.stack(xyz),
    contact=(np.stack(contact) if contact else np.zeros((0,), dtype=np.float32)),
    body_names=np.array(body_names), checkpoint=np.array(CK), family=np.array(FAMILY),
    pos_threshold=np.array(float(cmd.success_position_threshold)),
)
i150 = min(149, len(succ) - 1)
print(f"POSES wrote {OUT}: steps={len(succ)} envs={N} bodies={len(body_names)} "
      f"success@{i150 + 1}={succ[i150].mean():.3f} median_xyz={np.median(xyz[i150]):.4f}", flush=True)
env.close()
app.close()
