"""Render the ACTUAL ROBOT from recorded link poses, offscreen, without Isaac's renderer.

Isaac Sim's RTX renderer produces empty sky on this machine (six real defects were found and fixed in
the render path first; the camera is verifiably correct; the renderer reports `IOMMU is enabled` and
the stack wants driver branch 580 while the box runs 610). Physics was never affected.

So: `trace_full_poses.py` records position + orientation of every link each control step, the robot's
own visual meshes are pulled out of `ur10e_delto.usd`, and this poses those meshes and rasterises
them through Mesa's software EGL -- a path that never touches the NVIDIA driver that fails.

Every transform drawn here came from the physics engine. This is a rendering of the recorded motion,
not a re-simulation and not an artist's impression.

Usage:  render_meshes.py <poses_name> <out.mp4> [env_index|median|success]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

# pyrender 0.1.45 still calls np.infty, removed in NumPy 2.0, and it only fires when shadow mapping
# touches mesh.bounds -- so enabling shadows killed the render while the OLD mp4s stayed on disk and
# were nearly eye-checked as "the lit version". Restore the alias before pyrender is imported.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]

import pyrender  # noqa: E402
import trimesh  # noqa: E402

from pyrender.constants import RenderFlags  # noqa: E402

CS = Path("/home/mattephi/Work/Rubitek/cube-stack")
MESH_DIR = CS / "results" / "meshes"
W, H = int(os.environ.get("RENDER_W", "960")), int(os.environ.get("RENDER_H", "720"))
FPS = int(os.environ.get("RENDER_FPS", "15"))
STEP_SAMPLE = int(os.environ.get("RENDER_EVERY", "1"))

LINK_COLOUR = (243, 245, 248, 255)       # UR10e, light so the dark hand reads against it
HAND_COLOUR = (52, 58, 68, 255)          # DELTO dark
CUBE_COLOUR = (233, 78, 38, 255)         # carried cube -- the thing to watch
GOAL_COLOUR = (28, 108, 214, 255)        # receptive cube


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    # THE ORDER IS xyzw, NOT wxyz -- and this single mistake caused every rendering defect chased
    # for hours: the long arm links were flung across the frame while the near-identity hand links
    # looked correct, which sent me rebuilding meshes that were never wrong. Established by least
    # squares, not by reading docs: a rigid joint offset in the parent frame must reproduce the
    # observed parent->child world vector exactly. Over 5 joints x 150 steps x 64 envs,
    #     xyzw -> 0.000 mm residual on every joint
    #     wxyz -> 72-352 mm
    # Do not "fix" this back to wxyz without re-running that test.
    x, y, z, w = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def pose_mat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(q)
    T[:3, 3] = p
    return T


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = eye - target
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, f, eye
    return T


def main() -> None:
    name, out_path = sys.argv[1], sys.argv[2]
    pick = sys.argv[3] if len(sys.argv) > 3 else "median"
    d = np.load(CS / "results" / f"poses_{name}.npz", allow_pickle=True)
    body_pos, body_quat = d["body_pos"], d["body_quat"]        # (T, E, B, 3) / (T, E, B, 4)
    cube_pos, cube_quat = d["cube_pos"], d["cube_quat"]
    goal_pos, goal_quat = d["goal_pos"], d["goal_quat"]
    succ, xyz = d["success"], d["xyz_distance"]
    names = [str(x) for x in d["body_names"]]
    thr = float(d["pos_threshold"])
    T_steps, E, B = body_pos.shape[0], body_pos.shape[1], body_pos.shape[2]
    i150 = min(149, T_steps - 1)

    # WHICH ENVIRONMENT. Default to the MEDIAN by final error -- not the best, which would be a
    # flattering pick, and not env 0, which is arbitrary. "success" picks the median *among the
    # environments that met the task's own criterion*, which is the honest way to show a success:
    # a typical one, not the single prettiest.
    order = np.argsort(xyz[i150])
    if pick == "median":
        env = int(order[len(order) // 2])
    elif pick == "success":
        ok = [int(i) for i in order if succ[i150][i]]
        if not ok:
            print("NO SUCCESSFUL ENV -- falling back to median"); env = int(order[len(order) // 2])
        else:
            env = ok[len(ok) // 2]
    elif pick == "release":
        # For C3 the point of the clip is the RELEASE. Picking by "median final distance" chose an
        # environment whose fingers never opened (-3.2 mm) and whose cube left the pinch only because
        # the arm pulled away -- while the caption claimed a hand opening wide. Select instead among
        # environments where the cube is demonstrably held at step 0 and demonstrably gone by step 8,
        # and take the MEDIAN of those by how far the fingers open: typical, not the most dramatic.
        names_l = [str(x) for x in d["body_names"]]
        a, b = names_l.index("rl_dg_1_tip"), names_l.index("rl_dg_2_tip")
        tipmid = (body_pos[:, :, a] + body_pos[:, :, b]) / 2.0
        cdist = np.linalg.norm(cube_pos - tipmid, axis=2)
        fsep = np.linalg.norm(body_pos[:, :, a] - body_pos[:, :, b], axis=2)
        cand = [i for i in range(E) if cdist[0, i] < 0.030 and cdist[8, i] > 0.08]
        if not cand:
            print("NO CLEAR RELEASE ENV -- falling back to median"); env = int(order[len(order) // 2])
        else:
            cand.sort(key=lambda i: fsep[8, i] - fsep[0, i])
            env = int(cand[len(cand) // 2])
            print(f"RENDER release-pick: {len(cand)}/{E} envs held-then-released; "
                  f"median opening {(fsep[8, env] - fsep[0, env]) * 1000:.1f} mm, "
                  f"cube {cdist[0, env] * 1000:.0f} -> {cdist[8, env] * 1000:.0f} mm from the pinch")
    else:
        env = int(pick)
    print(f"RENDER {name}: env {env} of {E}, final xyz {xyz[i150][env]:.4f} m, "
          f"success={bool(succ[i150][env])}, run success {succ[i150].sum()}/{E}, threshold {thr} m")

    scene = pyrender.Scene(bg_color=[0.97, 0.975, 0.985, 1.0], ambient_light=[0.22, 0.22, 0.25])

    # Ground: a modest table-sized slab, not a 2.4 m wall. The old one filled half the frame and
    # dwarfed the 34 mm cubes that are the entire point of the picture.
    ground = trimesh.creation.box(extents=(1.1, 1.1, 0.02))
    ground.apply_translation([0.45, 0.02, -0.011])
    gmat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.42, 0.44, 0.47, 1.0], metallicFactor=0.0, roughnessFactor=0.85)
    scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False, material=gmat))

    nodes = {}
    missing = []
    for bi, bn in enumerate(names):
        f = MESH_DIR / f"{bn}.obj"
        if not f.exists():
            missing.append(bn)
            continue
        m = trimesh.load(f, process=False)
        col = HAND_COLOUR if bn.startswith("rl_dg") else LINK_COLOUR
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[c / 255.0 for c in col],
            metallicFactor=0.35 if bn.startswith("rl_dg") else 0.15,
            roughnessFactor=0.45 if bn.startswith("rl_dg") else 0.55)
        nodes[bi] = scene.add(pyrender.Mesh.from_trimesh(m, smooth=True, material=mat),
                              pose=pose_mat(body_pos[0, env, bi], body_quat[0, env, bi]))
    if missing:
        print(f"RENDER no mesh for {len(missing)} bodies (drawn as nothing): {missing}")
    print(f"RENDER {len(nodes)}/{B} link meshes posed")

    def prop(tag, colour, p, q):
        f = MESH_DIR / f"{tag}.obj"
        m = trimesh.load(f, process=False) if f.exists() else trimesh.creation.box(extents=(0.034,) * 3)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[c / 255.0 for c in colour], metallicFactor=0.05, roughnessFactor=0.7)
        return scene.add(pyrender.Mesh.from_trimesh(m, smooth=False, material=mat), pose=pose_mat(p, q))

    cube_node = prop("insertive_cube", CUBE_COLOUR, cube_pos[0, env], cube_quat[0, env])
    goal_node = prop("receptive_cube", GOAL_COLOUR, goal_pos[0, env], goal_quat[0, env])

    # TRACKING CAMERA. The previous camera was static and framed on the UNION of the whole
    # trajectory -- a 1.88 m hand path on C3 -- so the action (a box roughly 0.2 x 0.2 x 0.33 m around
    # the hand and the two cubes) rendered a few pixels wide, the arm was cropped at the elbow, and
    # the release that the caption described was invisible. Matte called the result garbage and was
    # right. This follows the smoothed hand/cube centroid at a fixed close radius instead.
    hand_ids = [i for i, n in enumerate(names) if n.startswith("rl_dg")]

    # WHAT THE CAMERA HOLDS ON, per family. Averaging the hand into the target makes the camera
    # follow the arm as it retreats -- on C3 that walked the goal cube off the bottom of frame, so the
    # viewer could not see where the cube was meant to end up. On C3 hold on the CUBE AND GOAL and let
    # the hand enter and leave; on C4 the hand is the action (it opens and withdraws), so include it.
    track_hand = "c3" not in name.lower()

    def focus_at(t):
        pts = [cube_pos[t, env][None, :], goal_pos[t, env][None, :]]
        if track_hand:
            pts.append(body_pos[t, env][hand_ids])
        return np.vstack(pts).mean(0)

    raw = np.stack([focus_at(t) for t in range(min(T_steps, 150))])
    # Smooth the target so the camera glides instead of jittering with the fingers.
    k = 21
    pad = np.pad(raw, ((k // 2, k // 2), (0, 0)), mode="edge")
    track = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(3)], axis=1)

    # Radius from the per-step extent of what must stay in frame, not from the whole path.
    # np.ptp(arr, axis=0), NOT arr.ptp(0): the ndarray method was removed in NumPy 2 and the
    # AttributeError killed the render silently behind an output filter that only printed lines
    # matching "RENDER" -- so stale 09:53 mp4s were eye-checked as though they were new output.
    def _ext_pts(t):
        pts = [cube_pos[t, env][None, :], goal_pos[t, env][None, :]]
        if track_hand:
            pts.append(body_pos[t, env][hand_ids])
        return np.vstack(pts)

    ext = np.stack([np.ptp(_ext_pts(t), axis=0) for t in range(min(T_steps, 150))])
    radius = float(max(np.linalg.norm(ext.max(0)) * 1.25, 0.55))
    if not track_hand:
        # C3: bias the aim point up toward the grasp height so the hand opening stays in frame, and
        # widen enough to keep both cubes AND the hand. Holding purely on the cubes framed the table
        # correctly but cropped the hand -- and the hand opening is the finding.
        hand_z = np.array([body_pos[t, env][hand_ids].mean(0) for t in range(min(T_steps, 150))])[:, 2]
        lift = float(np.clip(np.percentile(hand_z, 70) - track[:, 2].mean(), 0.0, 0.30))
        track = track + np.array([0.0, 0.0, lift * 0.65])
        radius = float(max(radius * 1.30, 0.70))
        print(f"RENDER C3 framing: lifted aim by {lift * 0.65:.3f} m, radius {radius:.3f} m "
              f"(keep both cubes AND the opening hand in frame)")
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.6, znear=0.01, zfar=40.0)
    cam_node = scene.add(cam, pose=look_at(track[0] + np.array([0.62, -0.70, 0.46]) * radius, track[0]))
    key_node = scene.add(pyrender.DirectionalLight(color=[1.0, 0.97, 0.92], intensity=4.2),
                         pose=look_at(track[0] + np.array([0.62, -0.70, 0.46]) * radius, track[0]))
    scene.add(pyrender.DirectionalLight(color=[0.85, 0.90, 1.0], intensity=2.0),
              pose=look_at(track[0] + np.array([-0.9, 0.75, 1.15]), track[0]))
    print(f"RENDER tracking camera: radius {radius:.3f} m, target {np.round(track[0], 3)} -> "
          f"{np.round(track[-1], 3)} (was a static camera on a {np.linalg.norm(np.diff(raw, axis=0), axis=1).sum():.2f} m path)")

    r = pyrender.OffscreenRenderer(W, H)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8, ffmpeg_log_level="error")
    # HOLD THE DECISIVE WINDOW. On C3 the entire finding -- the hand opening and the cube leaving --
    # happens in steps 0-5. At 15 fps that is a third of a second and simply cannot be seen, which is
    # why the clip did not show what its caption claimed. Repeat the early frames so the release plays
    # at roughly 1/5 speed, then run the rest at normal rate.
    SLOW_UNTIL, SLOW_FACTOR = 12, 5
    frames = []
    for t in range(0, min(T_steps, 150), STEP_SAMPLE):
        frames.extend([t] * (SLOW_FACTOR if t < SLOW_UNTIL else 1))
    stats = []
    for t in frames:
        for bi, nd in nodes.items():
            scene.set_pose(nd, pose_mat(body_pos[t, env, bi], body_quat[t, env, bi]))
        scene.set_pose(cube_node, pose_mat(cube_pos[t, env], cube_quat[t, env]))
        scene.set_pose(goal_node, pose_mat(goal_pos[t, env], goal_quat[t, env]))
        eye = track[t] + np.array([0.62, -0.70, 0.46]) * radius
        scene.set_pose(cam_node, look_at(eye, track[t]))
        scene.set_pose(key_node, look_at(eye, track[t]))
        col, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        wr.append_data(col)
        stats.append(float(col.std()))
    wr.close()
    r.delete()
    # A blank-frame check that cannot be fooled by uniform noise: per-frame spatial variation. A sky-
    # only frame has near-zero std; tonight a `std > 6` test certified 160 empty frames, so this also
    # reports the minimum rather than an average that one good frame could carry.
    sz = Path(out_path).stat().st_size
    print(f"RENDER wrote {out_path} ({len(frames)} frames, {sz / 1024:.0f} KB) "
          f"frame std min={min(stats):.2f} mean={np.mean(stats):.2f}")
    if min(stats) < 3.0:
        print("RENDER WARNING: at least one frame is nearly featureless -- LOOK at the contact sheet")


if __name__ == "__main__":
    main()
