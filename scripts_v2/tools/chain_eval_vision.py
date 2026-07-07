# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end box-assembly pipeline with VISION policies: chain diffusion RGB policies A->B->C.

Vision counterpart of ``chain_eval.py``. Where the state chainer loads three rsl_rl actors and
rebinds one observation manager per phase (the state obs is a function of entity poses), the
vision policies consume SCENE-FIXED camera images (front + side) plus proprioception, which are
stage-agnostic -- so there is NO obs rebinding. We simply swap which trained diffusion policy
drives the arm each phase, run synchronous A->B->C phases in one 4-entity scene, and use the SAME
geometric success check as chain_eval to decide when to advance.

Each ckpt_{a,b,c} is a diffusion_policy workspace checkpoint (step_0NNNNNN.ckpt / latest.ckpt)
trained via train_mlp_sim2real_image_with_aux_loss on that stage's RGB DataCollection dataset.

STATUS: scaffold -- authored while the RGB trainings run; validate on a matured vision ckpt.
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chain A->B->C box-assembly VISION (diffusion RGB) policies end-to-end.")
parser.add_argument("--task", type=str, default="OmniReset-Ur10eLinearGripper-BoxCenterPaper-RGB-Play-v0",
                    help="Base RGB-Play task (Stage A). Provides the front+side camera scene + robot/gripper.")
parser.add_argument("--num_envs", type=int, default=16, help="Cameras are heavy; keep modest.")
parser.add_argument("--ckpt_a", type=str, required=True, help="Stage-A diffusion RGB checkpoint (.ckpt).")
parser.add_argument("--ckpt_b", type=str, required=True, help="Stage-B diffusion RGB checkpoint (.ckpt).")
parser.add_argument("--ckpt_c", type=str, required=True, help="Stage-C diffusion RGB checkpoint (.ckpt).")
parser.add_argument("--steps_per_phase", type=int, default=320, help="Max control steps per stage before that env is failed.")
parser.add_argument("--max_stage", type=int, default=2, help="Highest stage to run (0=A only, 1=A+B, 2=A+B+C).")
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--hold_steps", type=int, default=5, help="Consecutive success steps required to advance a stage.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", type=str, default=None, help="If set, record a 3rd-person montage to this mp4.")
parser.add_argument("--rec_list", type=str, default="", help="Comma list of env ids to film (successful ones concatenated).")
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True  # RGB policies need rendered cameras
sys.argv = [sys.argv[0]] + remaining_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------------------------------------
import os

import gymnasium as gym
import numpy as np
import torch

import dill
import hydra
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import retrieve_file_path

from diffusion_policy.workspace.base_workspace import BaseWorkspace

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_assets import UWLAB_ASSETS_DATA_DIR
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as task_utils
from uwlab_tasks.utils.hydra import hydra_task_compose

# Per-pair metadata for the geometric success check (assembled_offset + receptive success thresholds).
_BOTTOM = f"{UWLAB_ASSETS_DATA_DIR}/Props/BoxAssembly/Bottom/bottom.usd"
_MID = f"{UWLAB_ASSETS_DATA_DIR}/Props/BoxAssembly/Mid/mid.usd"
_CAPRIM = f"{UWLAB_ASSETS_DATA_DIR}/Props/BoxAssembly/CapRim/caprim.usd"
_TARGET = f"{UWLAB_ASSETS_DATA_DIR}/Props/BoxAssembly/TableCenterTarget/target.usd"


def _offset(usd):
    m = task_utils.read_metadata_from_usd_directory(usd)
    ao = m.get("assembled_offset")
    st = m.get("success_thresholds") or {"position": 0.005, "orientation": 0.05}
    return tuple(ao["pos"]), tuple(ao["quat"]), float(st["position"]), float(st["orientation"])


def _load_policy(ckpt_path: str, device: torch.device):
    """Load a diffusion_policy workspace checkpoint -> eval policy (mirrors eval_distilled_policy)."""
    with open(retrieve_file_path(ckpt_path), "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.eval().to(device)


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):  # noqa: C901
    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
    N = args_cli.num_envs
    env_cfg.scene.num_envs = N
    env_cfg.sim.device = str(device)
    env_cfg.seed = args_cli.seed
    # obs as a dict of separate terms (rgb keys stay separate for the image policy)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.observations.policy.enable_corruption = False

    # 4-entity scene: the RGB Stage-A pair is insertive=Bottom(box), receptive=Target; augment adds
    # ctx_object(Mid) + ctx_cover(Cap). Override ctx_cover -> CapRim (the cover C was trained on).
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.box_assembly_aug import (
        augment_box_assembly,
    )

    augment_box_assembly(env_cfg, scene_only=True)
    env_cfg.scene.ctx_cover.spawn.usd_path = _CAPRIM

    if args_cli.video:
        env_cfg.scene.capture_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/capture_cam",
            update_period=0, height=540, width=720,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.16, -0.17, 0.56), rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948), convention="opengl"
            ),
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=20.0),
        )

    # We drive resets ourselves: drop the dataset reset (would replay A-only states, leaving ctx
    # entities unplaced) and stop episodes from auto-resetting mid-chain.
    if hasattr(env_cfg.events, "reset_from_reset_states"):
        env_cfg.events.reset_from_reset_states = None
    if hasattr(env_cfg.terminations, "abnormal_robot"):
        env_cfg.terminations.abnormal_robot = None
    env_cfg.episode_length_s = 1.0e6

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    u = env.unwrapped

    # --- load the three diffusion policies + per-policy obs-history wrappers ---
    pol_ckpts = {0: args_cli.ckpt_a, 1: args_cli.ckpt_b, 2: args_cli.ckpt_c}
    wrappers = {}
    for p, ck in pol_ckpts.items():
        pol = _load_policy(ck, device)
        wrappers[p] = DiffusionPolicyWrapper(pol, device, n_obs_steps=pol.n_obs_steps, num_envs=N)
    print("[chain-vision] loaded diffusion policies A/B/C")

    PAIRS = [("insertive_object", "receptive_object"), ("ctx_object", "insertive_object"), ("ctx_cover", "insertive_object")]

    # entity handles
    box = u.scene["insertive_object"]
    target = u.scene["receptive_object"]
    obj = u.scene["ctx_object"]
    cover = u.scene["ctx_cover"]

    offA_i = _offset(_BOTTOM); offA_r = _offset(_TARGET)
    offB_i = _offset(_MID); offB_r = _offset(_BOTTOM)
    offC_i = _offset(_CAPRIM); offC_r = _offset(_BOTTOM)
    PAIR = {
        0: (box, target, offA_i, offA_r, True),   # A: yaw-checked (canonical box)
        1: (obj, box, offB_i, offB_r, False),     # B
        2: (cover, box, offC_i, offC_r, False),   # C
    }

    def _apply_off(asset, off):
        p = asset.data.root_pos_w
        q = asset.data.root_quat_w
        op = torch.tensor(off[0], device=device).expand(N, 3)
        oq = torch.tensor(off[1], device=device).expand(N, 4)
        return math_utils.combine_frame_transforms(p, q, op, oq)

    def success(phase):
        ins, rec, off_i, off_r, check_yaw = PAIR[phase]
        ip, iq = _apply_off(ins, off_i)
        rp, rq = _apply_off(rec, off_r)
        rel_p, rel_q = math_utils.subtract_frame_transforms(rp, rq, ip, iq)
        ex, ey, ez = math_utils.euler_xyz_from_quat(rel_q)
        rpd = math_utils.wrap_to_pi(ex).abs() + math_utils.wrap_to_pi(ey).abs()
        yaw = (math_utils.wrap_to_pi(2.0 * math_utils.wrap_to_pi(ez)) / 2.0).abs()
        dist = torch.norm(rel_p, dim=1)
        pos_thr, ori_thr = off_r[2], off_r[3]
        ok = (dist < pos_thr) & (rpd < ori_thr)
        if check_yaw:
            ok = ok & (yaw < 0.0524)
        return ok, dist, rpd, yaw

    # --- custom reset: box resting (random yaw), object + cover resting nearby on the table ---
    origins = u.scene.env_origins  # (N,3)

    def _rand(a, b, n=1):
        return a + (b - a) * torch.rand(N, n, device=device)

    def place(asset, x_rng, y_rng, z, quat):
        pos = torch.zeros(N, 3, device=device)
        pos[:, 0] = _rand(*x_rng).squeeze(-1)
        pos[:, 1] = _rand(*y_rng).squeeze(-1)
        pos[:, 2] = z
        pos = pos + origins
        root = torch.cat([pos, quat], dim=1)
        asset.write_root_pose_to_sim(root)
        asset.write_root_velocity_to_sim(torch.zeros(N, 6, device=device))

    def yaw_quat(yaw):
        half = yaw / 2
        q = torch.zeros(N, 4, device=device)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    ident = torch.tensor([1.0, 0, 0, 0], device=device).expand(N, 4)
    TABLE_Z = -0.0135
    GOAL_XY = (0.45, 0.05)

    def reset_scene():
        env.reset()
        tpos = torch.zeros(N, 3, device=device)
        tpos[:, 0] = GOAL_XY[0]; tpos[:, 1] = GOAL_XY[1]; tpos[:, 2] = TABLE_Z
        target.write_root_pose_to_sim(torch.cat([tpos + origins, ident], dim=1))
        place(box, (0.39, 0.45), (0.08, 0.20), 0.03, yaw_quat(_rand(-3.1416, 3.1416).squeeze(-1)))
        place(obj, (0.52, 0.57), (-0.09, -0.02), 0.03, ident)
        place(cover, (0.52, 0.57), (0.22, 0.29), 0.03, ident)
        for _ in range(args_cli.settle_steps):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)
        u.observation_manager.reset()

    def get_obs():
        # fresh obs dict ({"policy": {term: tensor, ...}, ...}) after our manual placement/settle
        return u.observation_manager.compute()

    reset_scene()
    o0 = origins[0]
    print(f"[reset] (env0, rel-origin) box={(box.data.root_pos_w[0]-o0).tolist()} "
          f"target={(target.data.root_pos_w[0]-o0).tolist()} obj={(obj.data.root_pos_w[0]-o0).tolist()} "
          f"cover={(cover.data.root_pos_w[0]-o0).tolist()}")

    K = args_cli.hold_steps
    MAXP = args_cli.steps_per_phase
    cam = u.scene["capture_cam"] if args_cli.video else None
    rec_list = [int(x) for x in args_cli.rec_list.split(",") if x != ""] or [0]
    rec_list = [e for e in rec_list if e < N]
    frames_by_env = {e: [] for e in rec_list}

    def grab():
        u.sim.render()
        out = cam.data.output["rgb"]
        for e in rec_list:
            rgb = out[e, ..., :3].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            frames_by_env[e].append(rgb)

    all_ids = torch.arange(N, device=device)
    cleared = {0: torch.zeros(N, dtype=torch.bool, device=device),
               1: torch.zeros(N, dtype=torch.bool, device=device),
               2: torch.zeros(N, dtype=torch.bool, device=device)}

    with torch.inference_mode():
        for p, (ins, rec) in enumerate(PAIRS):
            if p > args_cli.max_stage:
                break
            wrappers[p].reset(all_ids)  # clear this policy's obs history + action queue
            obs_dict = get_obs()
            succ = torch.zeros(N, dtype=torch.long, device=device)
            for step in range(MAXP):
                action = wrappers[p].predict_action(obs_dict)
                step_out = env.step(action)
                obs_dict = step_out[0]
                if cam is not None:
                    grab()
                ok, dist, rpd, yaw = success(p)
                succ = torch.where(ok, succ + 1, torch.zeros_like(succ))
                cleared[p] = cleared[p] | (succ >= K)
                if step % 40 == 0 or step == MAXP - 1:
                    print(f"[stage {chr(65+p)} step {step:3d}] cleared={int(cleared[p].sum())}/{N} "
                          f"dist min/mean {dist.min():.3f}/{dist.mean():.3f} rpd {rpd.min():.3f} "
                          f"yaw {yaw.min():.3f} thr={PAIR[p][3][2]:.3f}")
                if cleared[p].all():
                    for _ in range(8):
                        step_out = env.step(wrappers[p].predict_action(obs_dict)); obs_dict = step_out[0]
                        if cam is not None:
                            grab()
                    break

    cA, cB, cC = cleared[0], cleared[1], cleared[2]
    e2e = (cA & cB & cC)
    print("\n============ VISION PIPELINE RESULT ============")
    print(f"envs={N}  task={args_cli.task}")
    print(f"stage A success:                 {int(cA.sum())}/{N} = {cA.float().mean():.3f}")
    print(f"stage B success | A:             {int((cA&cB).sum())}/{int(cA.sum()) or 1} = {(cB[cA].float().mean() if cA.any() else 0):.3f}")
    print(f"stage C success | A&B:           {int(e2e.sum())}/{int((cA&cB).sum()) or 1} = {(cC[cA&cB].float().mean() if (cA&cB).any() else 0):.3f}")
    print(f"END-TO-END success:              {int(e2e.sum())}/{N} = {e2e.float().mean():.3f}")
    print(f"marginal stage rates (all envs): A={cA.float().mean():.3f} B={cB.float().mean():.3f} C={cC.float().mean():.3f}  product={cA.float().mean()*cB.float().mean()*cC.float().mean():.3f}")
    print(f"e2e-success env ids: {e2e.nonzero(as_tuple=False).squeeze(-1).tolist()[:20]}")
    print("================================================")

    if args_cli.video and any(frames_by_env.values()):
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video)), exist_ok=True)
        import imageio.v2 as imageio
        goal = cleared[args_cli.max_stage]
        rank = sorted(rec_list, key=lambda e: (not bool(goal[e]), not bool(cA[e])))
        chosen = [e for e in rank if bool(goal[e])] or rank
        seq = []
        for e in chosen:
            seq.extend(frames_by_env[e])
        imageio.mimwrite(args_cli.video, seq, fps=args_cli.fps, macro_block_size=None)
        print(f"[chain-vision] wrote {len(seq)} frames from envs {chosen} "
              f"({sum(bool(e2e[e]) for e in chosen)} e2e-success) -> {args_cli.video}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
