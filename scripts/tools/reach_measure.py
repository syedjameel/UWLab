"""Measure where the UR5e+DELTO hand, the object and the goal ACTUALLY are at reset,
and sweep the arm's reachable palm set by forward kinematics through the articulation.

Single-shot diagnostic. Writes JSON to --out.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="DexLift-UR5eDelto-RelJointPos-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--fk_iters", type=int, default=4000)
parser.add_argument("--out", type=str, default="/root/reach_measure.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT: dict = {}


def stats(t: torch.Tensor) -> dict:
    t = t.float().flatten()
    return {
        "min": float(t.min()),
        "p25": float(t.quantile(0.25)),
        "median": float(t.median()),
        "p75": float(t.quantile(0.75)),
        "max": float(t.max()),
        "mean": float(t.mean()),
    }


def box(p: torch.Tensor) -> dict:
    return {
        "x": [float(p[:, 0].min()), float(p[:, 0].max())],
        "y": [float(p[:, 1].min()), float(p[:, 1].max())],
        "z": [float(p[:, 2].min()), float(p[:, 2].max())],
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    uenv.reset()

    scene = uenv.scene
    robot = scene["robot"]
    obj = scene["object"]
    origins = scene.env_origins  # (N,3)

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    OUT["body_names"] = body_names
    OUT["joint_names"] = joint_names

    # -- physics gravity actually in force at reset
    try:
        g = uenv.sim.get_physics_context().get_gravity()
        OUT["gravity_at_reset"] = [float(x) for x in (g if isinstance(g, (list, tuple)) else [g])]
    except Exception as e:  # noqa: BLE001
        try:
            pc = uenv.sim.get_physics_context()
            OUT["gravity_at_reset"] = {
                "direction": [float(x) for x in pc.get_gravity_direction()],
                "magnitude": float(pc.get_gravity_magnitude()),
            }
        except Exception as e2:  # noqa: BLE001
            OUT["gravity_at_reset"] = f"UNREADABLE: {e} / {e2}"

    tip_ids, tip_names = robot.find_bodies(".*tip.*")
    palm_ids, palm_names = robot.find_bodies("rl_dg_mount")
    if not palm_ids:
        palm_ids, palm_names = robot.find_bodies(".*mount.*")
    OUT["fingertip_bodies"] = tip_names
    OUT["palm_body"] = palm_names

    bp = robot.data.body_pos_w  # (N, B, 3)
    tips_w = bp[:, tip_ids, :]
    palm_w = bp[:, palm_ids[0], :]
    root_w = robot.data.root_pos_w
    obj_w = obj.data.root_pos_w

    tips_e = tips_w - origins[:, None, :]
    palm_e = palm_w - origins
    root_e = root_w - origins
    obj_e = obj_w - origins

    OUT["reset"] = {}
    OUT["reset"]["robot_root_env_frame_mean"] = [float(v) for v in root_e.mean(0)]
    OUT["reset"]["palm_env_frame_mean"] = [float(v) for v in palm_e.mean(0)]
    OUT["reset"]["palm_env_frame_std"] = [float(v) for v in palm_e.std(0)]
    OUT["reset"]["palm_box"] = box(palm_e)
    OUT["reset"]["fingertips_env_frame_mean"] = {
        n: [float(v) for v in tips_e[:, i, :].mean(0)] for i, n in enumerate(tip_names)
    }
    OUT["reset"]["object_env_frame_mean"] = [float(v) for v in obj_e.mean(0)]
    OUT["reset"]["object_box"] = box(obj_e)
    OUT["reset"]["object_z_stats"] = stats(obj_e[:, 2])

    d_tip_obj = torch.linalg.norm(tips_e - obj_e[:, None, :], dim=-1)  # (N, T)
    nearest = d_tip_obj.min(dim=1).values
    OUT["reset"]["dist_object_to_nearest_fingertip"] = stats(nearest)
    OUT["reset"]["dist_object_to_each_fingertip_mean"] = {
        n: float(d_tip_obj[:, i].mean()) for i, n in enumerate(tip_names)
    }
    OUT["reset"]["dist_object_to_palm"] = stats(torch.linalg.norm(obj_e - palm_e, dim=-1))
    OUT["reset"]["dist_object_to_robot_base"] = stats(torch.linalg.norm(obj_e - root_e, dim=-1))
    OUT["reset"]["dist_palm_to_robot_base"] = stats(torch.linalg.norm(palm_e - root_e, dim=-1))

    # -- commanded goal
    try:
        term = uenv.command_manager.get_term("object_pose")
        goal_w = term.pose_command_w[:, :3]
        goal_e = goal_w - origins
        OUT["reset"]["goal_env_frame_mean"] = [float(v) for v in goal_e.mean(0)]
        OUT["reset"]["goal_box"] = box(goal_e)
        OUT["reset"]["dist_goal_to_palm"] = stats(torch.linalg.norm(goal_e - palm_e, dim=-1))
        OUT["reset"]["dist_goal_to_object"] = stats(torch.linalg.norm(goal_e - obj_e, dim=-1))
        OUT["reset"]["dist_goal_to_robot_base"] = stats(torch.linalg.norm(goal_e - root_e, dim=-1))
    except Exception as e:  # noqa: BLE001
        OUT["reset"]["goal"] = f"UNREADABLE: {e}"

    # -- arm reset joint configuration
    arm_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    arm_ids, arm_found = robot.find_joints(arm_names, preserve_order=True)
    q = robot.data.joint_pos
    OUT["arm_reset_joint_pos_rad"] = {n: float(q[:, i].mean()) for n, i in zip(arm_found, arm_ids)}
    OUT["arm_reset_joint_pos_std"] = {n: float(q[:, i].std()) for n, i in zip(arm_found, arm_ids)}
    OUT["arm_default_joint_pos_rad"] = {
        n: float(robot.data.default_joint_pos[0, i]) for n, i in zip(arm_found, arm_ids)
    }
    lim = robot.data.joint_pos_limits  # (N, J, 2)
    OUT["arm_joint_limits_rad"] = {
        n: [float(lim[0, i, 0]), float(lim[0, i, 1])] for n, i in zip(arm_found, arm_ids)
    }
    soft = robot.data.soft_joint_pos_limits
    OUT["arm_soft_joint_limits_rad"] = {
        n: [float(soft[0, i, 0]), float(soft[0, i, 1])] for n, i in zip(arm_found, arm_ids)
    }
    OUT["hand_reset_joint_pos_rad"] = {
        n: float(q[:, i].mean()) for i, n in enumerate(joint_names) if i not in arm_ids
    }

    # -- FK sweep of the arm's reachable palm set
    n_env = uenv.num_envs
    dev = uenv.device
    lo = lim[0, arm_ids, 0].clone()
    hi = lim[0, arm_ids, 1].clone()
    q_full = robot.data.joint_pos.clone()
    zero_v = torch.zeros_like(q_full)
    palm_id = palm_ids[0]

    samples = []
    tip_samples = []
    q_samples = []
    dt = uenv.sim.get_physics_dt()
    for _ in range(args_cli.fk_iters):
        r = torch.rand((n_env, len(arm_ids)), device=dev)
        qa = lo + r * (hi - lo)
        q_full[:, arm_ids] = qa
        robot.write_joint_state_to_sim(q_full, zero_v)
        robot.set_joint_position_target(q_full)
        robot.write_data_to_sim()
        uenv.sim.step(render=False)
        robot.update(dt)
        samples.append((robot.data.body_pos_w[:, palm_id, :] - origins).clone())
        tip_samples.append((robot.data.body_pos_w[:, tip_ids, :] - origins[:, None, :]).clone())
        q_samples.append(qa.clone())

    P = torch.cat(samples, dim=0)  # (M,3) palm positions, env frame
    T = torch.cat(tip_samples, dim=0)  # (M,T,3)
    Q = torch.cat(q_samples, dim=0)
    M = P.shape[0]
    OUT["fk"] = {"n_samples": int(M)}

    # verification: FK write actually took effect
    resid = float((Q[-n_env:] - robot.data.joint_pos[:, arm_ids]).abs().max())
    OUT["fk"]["max_joint_write_residual_rad"] = resid

    OUT["fk"]["palm_box_env_frame"] = box(P)
    base = root_e.mean(0)
    OUT["fk"]["robot_base_env_frame"] = [float(v) for v in base]
    rad = torch.linalg.norm(P - base, dim=-1)
    OUT["fk"]["palm_radius_from_base"] = stats(rad)
    OUT["fk"]["palm_radius_p99"] = float(rad.quantile(0.99))

    tip_rad = torch.linalg.norm(T - base, dim=-1).max(dim=1).values
    OUT["fk"]["farthest_fingertip_radius_from_base"] = stats(tip_rad)
    OUT["fk"]["fingertip_box_env_frame"] = box(T.reshape(-1, 3))

    # planar reach at the workspace height band
    for zlo, zhi, tag in [(0.0, 0.10, "z_0.00_0.10"), (0.0, 0.30, "z_0.00_0.30")]:
        m = (P[:, 2] >= zlo) & (P[:, 2] <= zhi)
        if int(m.sum()) > 0:
            OUT["fk"][f"palm_xy_radius_{tag}"] = stats(torch.linalg.norm(P[m][:, :2] - base[:2], dim=-1))
            OUT["fk"][f"palm_x_max_{tag}"] = float(P[m][:, 0].max())

    # -- fraction of the OmniReset envelope reachable by the PALM
    gx = torch.arange(0.35, 0.6001, 0.01, device=dev)
    gy = torch.arange(-0.20, 0.2001, 0.01, device=dev)
    gz = torch.arange(0.00, 0.3001, 0.01, device=dev)
    G = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1).reshape(-1, 3)
    mind = torch.full((G.shape[0],), 1e9, device=dev)
    chunk = 20000
    for i in range(0, M, chunk):
        d = torch.cdist(G, P[i : i + chunk])
        mind = torch.minimum(mind, d.min(dim=1).values)
    env_res = {"n_grid_points": int(G.shape[0]), "grid_spacing_m": 0.01}
    for tol in (0.01, 0.02, 0.03, 0.05):
        env_res[f"frac_reachable_tol_{tol}"] = float((mind <= tol).float().mean())
    env_res["min_dist_stats"] = stats(mind)
    reach = mind <= 0.02
    if int(reach.sum()) > 0:
        env_res["reachable_subbox_tol_0.02"] = box(G[reach])
    unreach = ~reach
    if int(unreach.sum()) > 0:
        env_res["unreachable_subbox_tol_0.02"] = box(G[unreach])
        # where does the unreachable part live
        env_res["unreachable_x_stats"] = stats(G[unreach][:, 0])
        env_res["unreachable_z_stats"] = stats(G[unreach][:, 2])
    OUT["omnireset_envelope_palm"] = env_res

    # -- same envelope, but "can a FINGERTIP get there" (what actually matters for contact)
    Tf = T.reshape(-1, 3)
    mind2 = torch.full((G.shape[0],), 1e9, device=dev)
    for i in range(0, Tf.shape[0], chunk):
        d = torch.cdist(G, Tf[i : i + chunk])
        mind2 = torch.minimum(mind2, d.min(dim=1).values)
    env2 = {}
    for tol in (0.01, 0.02, 0.03, 0.05):
        env2[f"frac_reachable_tol_{tol}"] = float((mind2 <= tol).float().mean())
    env2["min_dist_stats"] = stats(mind2)
    OUT["omnireset_envelope_fingertip"] = env2

    # -- fraction of the ACTUAL dexsuite object-spawn volume that is palm-reachable
    ob = OUT["reset"]["object_box"]
    ax = torch.arange(0.35, 0.7501, 0.01, device=dev)
    ay = torch.arange(-0.20, 0.2001, 0.01, device=dev)
    az = torch.arange(0.09, 0.5001, 0.01, device=dev)
    Ga = torch.stack(torch.meshgrid(ax, ay, az, indexing="ij"), dim=-1).reshape(-1, 3)
    minda = torch.full((Ga.shape[0],), 1e9, device=dev)
    for i in range(0, M, chunk):
        d = torch.cdist(Ga, P[i : i + chunk])
        minda = torch.minimum(minda, d.min(dim=1).values)
    OUT["dexsuite_spawn_volume_palm"] = {
        "volume_box": {"x": [0.35, 0.75], "y": [-0.2, 0.2], "z": [0.09, 0.50]},
        "observed_object_box": ob,
        "frac_reachable_tol_0.02": float((minda <= 0.02).float().mean()),
        "frac_reachable_tol_0.05": float((minda <= 0.05).float().mean()),
        "min_dist_stats": stats(minda),
    }

    with open(args_cli.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print("WROTE", args_cli.out)
    print(json.dumps(OUT, indent=2))
    env.close()


main()
print("SCRIPT_COMPLETE")
simulation_app.close()
