# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box-assembly task specifics for OmniReset.

Three sequential placement stages share ONE scene that always spawns all four
entities (box / object / cover / assembly-target). This module provides:

* ``resolve_entities`` -- map the semantic identities (box, object, cover, target)
  to the scene-entity names actually present (insertive_object / receptive_object /
  the ``ctx_*`` context entities), keyed on each asset's USD path.
* ``scatter_objects_no_contact`` -- reset EventTerm that drops the *free* context
  objects onto the table at random poses, optionally enforcing a minimum
  separation from every other object (the ``allow_overlap`` toggle, spawn rule 1c).
* ``box_assembly_goal`` -- a ``ProgressContext``-compatible reward/term that
  verifies the *cumulative* geometric goal (rule 2.1): each stage's success
  requires the current placement AND all preceding placements to hold, plus a
  workspace constraint (rule 2.2). Geometric checks are XY / height / yaw within
  configurable margins (default 3 mm / 6 mm / 3 deg, spec note 3).

Asset geometry mirrors ``scripts_v2/tools/author_box_assembly_assets.py`` (the
MuJoCo ``scene.xml`` collision boxes), so footprints / rects used for the
"fully inside" containment match the authored USDs exactly.
"""

from __future__ import annotations

import math
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..assembly_keypoints import Offset
from . import utils

__all__ = [
    "USD_FOLDER_TO_SEMANTIC",
    "SEMANTIC_TO_USD_FOLDER",
    "GEOM",
    "STAGE_CHAINS",
    "box_assembly_asset_dir",
    "asset_usd",
    "semantic_of_usd",
    "resolve_entities",
    "scatter_objects_no_contact",
    "box_assembly_goal",
]

# ---------------------------------------------------------------------------------------------------
# Semantic identities <-> authored USD folder names (object_name_from_usd output).
# ---------------------------------------------------------------------------------------------------
USD_FOLDER_TO_SEMANTIC = {
    "Bottom": "box",
    "Mid": "object",
    "Cap": "cover",
    # Cover ablation lids (no-knob / edge-rim) are also semantically the "cover": this makes the
    # augment recognise them as the cover-close insertive (so the object is spawned INSIDE the box).
    "CapNoKnob": "cover",
    "CapRim": "cover",
    "TableCenterTarget": "target",
}
# Explicit inverse (NOT auto-derived): the canonical cover USD spawned for a *context* cover (stages
# A/B) stays "Cap", so the ablation lids do not change A/B. They only ever appear as the insertive in
# their own cover-close task, where the cover USD is set explicitly.
SEMANTIC_TO_USD_FOLDER = {"box": "Bottom", "object": "Mid", "cover": "Cap", "target": "TableCenterTarget"}

# Half-extents / rects in each object's local frame (metres), transcribed from the authoring tool.
GEOM = {
    # box (Bottom): outer footprint and the inner cavity rect (between the tray walls).
    "box_outer_half": (0.040, 0.025),
    "box_inner_half": (0.035, 0.020),  # inside the 2.5 mm walls
    "box_floor_top_z": 0.005,
    "box_wall_top_z": 0.020,
    # object (Mid): solid box half extents.
    "object_half": (0.0272, 0.0152, 0.009),
    # cover (Cap): outer footprint and the inner rect the box seats into.
    "cover_outer_half": (0.0275, 0.0275),
    "cover_inner_half": (0.023, 0.023),
    # assembly target: the on-table marker rect the box must sit fully inside.
    "target_half": (0.045, 0.030),
}

# Per-stage cumulative goal chain: list of (insertive_semantic, receptive_semantic).
# The LAST entry is the current stage; earlier entries are preceding-stage goals that
# must still hold (rule 2.1).
STAGE_CHAINS = {
    "box_center": [("box", "target")],
    "object_in_box": [("box", "target"), ("object", "box")],
    "cover_close": [("box", "target"), ("object", "box"), ("cover", "box")],
}


def box_assembly_asset_dir() -> str:
    from uwlab_assets import UWLAB_ASSETS_DATA_DIR

    return f"{UWLAB_ASSETS_DATA_DIR}/Props/BoxAssembly"


def asset_usd(semantic: str) -> str:
    folder = SEMANTIC_TO_USD_FOLDER[semantic]
    fname = "target.usd" if semantic == "target" else f"{folder.lower()}.usd"
    return f"{box_assembly_asset_dir()}/{folder}/{fname}"


def semantic_of_usd(usd_path: str) -> str | None:
    """Return the semantic id (box/object/cover/target) for a USD path, or None."""
    return USD_FOLDER_TO_SEMANTIC.get(utils.object_name_from_usd(usd_path))


def resolve_entities(env: ManagerBasedEnv) -> dict[str, str]:
    """Map semantic id -> scene-entity name for every box-assembly rigid object present."""
    mapping: dict[str, str] = {}
    for name, obj in env.scene.rigid_objects.items():
        usd_path = getattr(getattr(obj.cfg, "spawn", None), "usd_path", None)
        if usd_path is None:
            continue
        sem = semantic_of_usd(usd_path)
        if sem is not None:
            mapping[sem] = name
    return mapping


# ---------------------------------------------------------------------------------------------------
# Spawn: scatter the free context objects (spawn rules 1b/1c).
# ---------------------------------------------------------------------------------------------------
class scatter_objects_no_contact(ManagerTermBase):
    """Reset the *free* context objects to random table poses.

    When ``allow_overlap`` is False (default), rejection-sample positions so every
    scattered object is at least ``min_separation`` from every other scattered object
    AND from each ``avoid_cfgs`` object (the work objects placed by earlier events) --
    enforcing the no-contact spawn rule cheaply (position-only, no SDF). When True, the
    separation constraint is dropped, so objects may pile up (the hard scenario, 1c).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.object_cfgs: list[SceneEntityCfg] = cfg.params.get("object_cfgs", [])
        self.avoid_cfgs: list[SceneEntityCfg] = cfg.params.get("avoid_cfgs", [])
        pose_range = cfg.params.get("pose_range", {})
        self.ranges = torch.tensor(
            [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "yaw"]], device=env.device
        )
        self.use_bottom_offset = cfg.params.get("use_bottom_offset", True)
        # same table-surface datum the validated resets use (ur5_metal_support default z = -0.013);
        # without it scattered objects spawn ~13 mm above the table and free-fall on reset.
        self.offset_asset_cfg = cfg.params.get("offset_asset_cfg", SceneEntityCfg("ur5_metal_support"))
        # per-object base (roll, pitch) applied before the random yaw -- used in easy mode to spawn
        # objects in their correct upright orientation (box cavity-up, cover lid-down). {} = flat.
        self.base_euler = cfg.params.get("base_euler", {})
        if self.use_bottom_offset:
            self.bottom_offsets = {}
            for ocfg in self.object_cfgs:
                meta = utils.read_metadata_from_usd_directory(env.scene[ocfg.name].cfg.spawn.usd_path)
                self.bottom_offsets[ocfg.name] = float(meta.get("bottom_offset", {}).get("pos", [0, 0, 0])[2])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        object_cfgs: list[SceneEntityCfg],
        avoid_cfgs: list[SceneEntityCfg] = [],
        pose_range: dict = {},
        min_separation: float = 0.10,
        allow_overlap: bool = False,
        use_bottom_offset: bool = True,
        offset_asset_cfg: SceneEntityCfg | None = SceneEntityCfg("ur5_metal_support"),
        base_euler: dict | None = None,
        max_tries: int = 12,
    ) -> None:
        n = len(env_ids)
        origins = env.scene.env_origins[env_ids]
        # table-surface datum: rest objects on the support plate, not the env-origin plane.
        offset_z = torch.zeros((n, 1), device=env.device)
        if self.offset_asset_cfg is not None:
            offset_z = env.scene[self.offset_asset_cfg.name].data.default_root_state[env_ids, 2:3]

        # centres to avoid: the work objects already placed this reset (relative to origin).
        avoid = []
        for acfg in avoid_cfgs:
            asset: RigidObject = env.scene[acfg.name]
            avoid.append((asset.data.root_pos_w[env_ids] - origins)[:, :2])
        avoid_xy = torch.stack(avoid, dim=1) if avoid else torch.zeros((n, 0, 2), device=env.device)

        placed_xy = avoid_xy  # grows as we place each free object
        for ocfg in self.object_cfgs:
            asset = env.scene[ocfg.name]
            lo, hi = self.ranges[:, 0], self.ranges[:, 1]
            samp = math_utils.sample_uniform(lo, hi, (n, 4), device=env.device)
            xy = samp[:, :2]
            if not allow_overlap and placed_xy.shape[1] > 0:
                for _ in range(max_tries):
                    d = torch.cdist(xy.unsqueeze(1), placed_xy).squeeze(1)  # (n, k)
                    bad = (d < min_separation).any(dim=1)
                    if not bad.any():
                        break
                    resamp = math_utils.sample_uniform(lo, hi, (int(bad.sum()), 4), device=env.device)
                    xy = xy.clone()
                    xy[bad] = resamp[:, :2]
            # rest the object on the table: lift origin by bottom_offset so its lowest
            # collision point sits at table z; pose_range z is small jitter on top.
            z = samp[:, 2:3] + offset_z
            if self.use_bottom_offset:
                z = z + self.bottom_offsets.get(ocfg.name, 0.0)
            pos = torch.cat([xy, z], dim=1) + origins
            yaw = samp[:, 3]
            z0 = torch.zeros_like(yaw)
            yaw_q = math_utils.quat_from_euler_xyz(z0, z0, yaw)
            roll, pitch = self.base_euler.get(ocfg.name, (0.0, 0.0))
            if roll or pitch:
                base_q = math_utils.quat_from_euler_xyz(
                    torch.full_like(yaw, float(roll)), torch.full_like(yaw, float(pitch)), z0
                )
                quat = math_utils.quat_mul(yaw_q, base_q)  # yaw about world-z, keeping the base tilt
            else:
                quat = yaw_q
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
            placed_xy = torch.cat([placed_xy, xy.unsqueeze(1)], dim=1)


# ---------------------------------------------------------------------------------------------------
# Verify: cumulative geometric goal (rules 2.1, 2.2, 2.4-2.6).
# ---------------------------------------------------------------------------------------------------
def _fold_yaw(e_z: torch.Tensor) -> torch.Tensor:
    """Yaw error folded modulo pi (box / object / cover all have 180-deg footprint symmetry)."""
    y = math_utils.wrap_to_pi(e_z).abs()
    return torch.minimum(y, (math.pi - y).abs())


class box_assembly_goal(ManagerTermBase):
    """``ProgressContext``-compatible cumulative geometric goal checker.

    Exposes ``success``, ``position_aligned``, ``orientation_aligned``,
    ``xyz_distance`` and ``euler_xy_distance`` so the shipped ``success_reward`` /
    ``dense_success_reward`` / ``MultiResetManager`` success hooks work unchanged when
    this term is registered under the name ``progress_context``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.stage: str = cfg.params.get("stage")
        self.xy_tol: float = cfg.params.get("xy_tol", 0.003)
        self.z_tol: float = cfg.params.get("z_tol", 0.006)
        self.yaw_tol: float = cfg.params.get("yaw_tol", math.radians(3.0))
        self.rp_tol: float = cfg.params.get("rp_tol", 0.10)
        # Ablation toggles (defaults preserve shipped behavior). The reference ProgressContext
        # checks neither yaw nor a workspace AABB; set these False to match it for fair RL comparison.
        self.check_yaw: bool = bool(cfg.params.get("check_yaw", True))
        self.use_workspace_gate: bool = bool(cfg.params.get("use_workspace_gate", True))
        # workspace AABB (relative to env origin) -- objects ejected from here fail (rule 2.2).
        # Compact region around the work zone (about half the earlier span); matches the tightened
        # spawn ranges in box_assembly_aug so spawned objects start inside the workspace.
        self.ws_xy = cfg.params.get("workspace_xy", ((0.28, 0.66), (-0.20, 0.40)))
        self.ws_z = cfg.params.get("workspace_z", (-0.05, 0.45))

        ent = resolve_entities(env)
        self._chain = []
        for ins_sem, rec_sem in STAGE_CHAINS[self.stage]:
            ins = env.scene[ent[ins_sem]]
            rec = env.scene[ent[rec_sem]]
            ins_meta = utils.read_metadata_from_usd_directory(ins.cfg.spawn.usd_path)
            rec_meta = utils.read_metadata_from_usd_directory(rec.cfg.spawn.usd_path)
            self._chain.append((
                ins,
                rec,
                Offset(pos=tuple(ins_meta["assembled_offset"]["pos"]), quat=tuple(ins_meta["assembled_offset"]["quat"])),
                Offset(pos=tuple(rec_meta["assembled_offset"]["pos"]), quat=tuple(rec_meta["assembled_offset"]["quat"])),
            ))
        # the three physical objects whose workspace membership we guard
        self._objects = [env.scene[ent[s]] for s in ("box", "object", "cover") if s in ent]

        z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.success = z.clone()
        self.position_aligned = z.clone()
        self.orientation_aligned = z.clone()
        self.xyz_distance = torch.zeros(env.num_envs, device=env.device)
        self.euler_xy_distance = torch.zeros(env.num_envs, device=env.device)
        self.continuous_success_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def _relation(self, ins, rec, ins_off, rec_off):
        ins_pos, ins_quat = ins_off.apply(ins)
        rec_pos, rec_quat = rec_off.apply(rec)
        rel_pos, rel_quat = math_utils.subtract_frame_transforms(rec_pos, rec_quat, ins_pos, ins_quat)
        xy = torch.norm(rel_pos[:, :2], dim=1)
        z = rel_pos[:, 2].abs()
        e_x, e_y, e_z = math_utils.euler_xyz_from_quat(rel_quat)
        rp = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        yaw = _fold_yaw(e_z)
        pos_ok = (xy < self.xy_tol) & (z < self.z_tol)
        ori_ok = rp < self.rp_tol
        if self.check_yaw:
            ori_ok = ori_ok & (yaw < self.yaw_tol)
        return pos_ok, ori_ok, xy + z, rp + yaw

    def in_workspace(self) -> torch.Tensor:
        (x0, x1), (y0, y1) = self.ws_xy
        z0, z1 = self.ws_z
        ok = torch.ones(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        for obj in self._objects:
            p = obj.data.root_pos_w - self._env.scene.env_origins
            ok &= (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] > z0) & (p[:, 2] < z1)
        return ok

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self.continuous_success_counter[:] = 0
        else:
            self.continuous_success_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stage: str | None = None,
        check_yaw: bool = True,
        use_workspace_gate: bool = True,
        xy_tol: float = 0.003,
        z_tol: float = 0.006,
        yaw_tol: float = math.radians(3.0),
        rp_tol: float = 0.10,
    ) -> torch.Tensor:
        # All params are resolved/stored in __init__; the call-time args are unused but must each be
        # named explicitly here (IsaacLab statically validates reward-term params against THIS
        # signature, not __init__; **kwargs would fail, and any param in the cfg missing here errors).
        pos_all = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        ori_all = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for i, (ins, rec, ins_off, rec_off) in enumerate(self._chain):
            pos_ok, ori_ok, xyz, exy = self._relation(ins, rec, ins_off, rec_off)
            pos_all &= pos_ok
            ori_all &= ori_ok
            if i == len(self._chain) - 1:  # current stage drives dense-reward shaping signals
                self.xyz_distance[:] = xyz
                self.euler_xy_distance[:] = exy
        ws = self.in_workspace() if self.use_workspace_gate else True
        self.position_aligned[:] = pos_all & ws
        self.orientation_aligned[:] = ori_all
        self.success[:] = self.position_aligned & self.orientation_aligned
        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )
        return torch.zeros(env.num_envs, device=env.device)
