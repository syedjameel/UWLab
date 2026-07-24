# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Eyeball recorded RGB demos from a collect_demos zarr (diffusion-policy ReplayBuffer layout).

Reads ``data/obs/*_rgb`` + ``meta/episode_ends`` and writes, per sampled episode, an MP4
with all cameras side by side, plus one ``contact_sheet.png`` (rows = episodes, cols =
timesteps of the front camera) for a single-glance overview. Plain python -- no Isaac:

    python scripts_v2/tools/visualize_rgb_demos.py \
        --dataset datasets/ur10e_pcb/rgb_smoke.zarr --out demo_viz --episodes 8

Review checklist (sim2real doc 10.4): wrist camera tracks the gripper (never a frozen /
black view), mats + curtains + objects retexture across episodes, lighting varies,
cameras framed like the real captures, the demos actually finish the assembly.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import cv2
except ImportError as exc:  # cv2 ships with the isaac envs; guard for bare envs
    raise SystemExit(f"needs opencv-python (cv2): {exc}")

import zarr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Path to the .zarr written by collect_demos.py")
    ap.add_argument("--out", default="demo_viz", help="Output directory for MP4s + contact sheet.")
    ap.add_argument("--episodes", type=int, default=8, help="How many episodes to export (spread over the dataset).")
    ap.add_argument("--fps", type=int, default=10, help="MP4 playback fps (collection runs at 10 Hz).")
    ap.add_argument("--sheet_cols", type=int, default=8, help="Timesteps per episode row in the contact sheet.")
    args = ap.parse_args()

    root = zarr.open(args.dataset, mode="r")
    ends = np.asarray(root["meta/episode_ends"])
    obs = root["data"]["obs"] if "obs" in root["data"] else root["data"]
    cams = sorted(k for k in obs.keys() if k.endswith("_rgb"))
    if not cams:
        raise SystemExit(f"no *_rgb keys under data/obs -- found: {list(obs.keys())}")
    n_eps = len(ends)
    print(f"[viz] {args.dataset}: {n_eps} episodes, {ends[-1]} steps, cameras: {cams}")
    lengths = np.diff(np.concatenate([[0], ends]))
    print(f"[viz] episode length: min {lengths.min()} / mean {lengths.mean():.1f} / max {lengths.max()}")

    os.makedirs(args.out, exist_ok=True)

    # ---- anomaly scan over ALL episodes (subsampled frames) ----
    # Flat frames (std < 10, the corrupted_camera signature) AND noise-like frames
    # (std > 80; textured scenes sit ~30-70, uniform random noise ~74+) both indicate
    # renderer corruption. Flagged episodes are added to the MP4 export for eyeballing.
    flagged: dict[int, str] = {}
    for ep in range(n_eps):
        s, e = (0 if ep == 0 else int(ends[ep - 1])), int(ends[ep])
        ts = np.linspace(s, e - 1, min(6, e - s)).astype(int)
        for c in cams:
            stds = np.asarray([np.asarray(obs[c][int(t)]).std() for t in ts])
            if (stds < 10.0).any():
                flagged[ep] = f"{c}: FLAT frame (min std {stds.min():.1f})"
            elif (stds > 80.0).any():
                flagged[ep] = f"{c}: NOISE-like frame (max std {stds.max():.1f})"
    if flagged:
        print(f"[viz] SUSPECT episodes ({len(flagged)}/{n_eps}):")
        for ep, why in sorted(flagged.items()):
            print(f"[viz]   episode {ep}: {why}")
    else:
        print(f"[viz] anomaly scan clean: no flat/noise frames in {n_eps} episodes")

    picks = np.unique(
        np.concatenate([
            np.linspace(0, n_eps - 1, min(args.episodes, n_eps)).astype(int),
            np.array(sorted(flagged)[:10], dtype=int),
        ])
    )

    sheet_rows = []
    for ep in picks:
        s, e = (0 if ep == 0 else int(ends[ep - 1])), int(ends[ep])
        frames = {c: np.asarray(obs[c][s:e]) for c in cams}  # (T, H, W, 3) uint8
        T = e - s
        first = frames[cams[0]]
        h, w = first.shape[1:3]

        path = os.path.join(args.out, f"episode_{ep:04d}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * len(cams), h))
        for t in range(T):
            row = np.concatenate([frames[c][t] for c in cams], axis=1)
            vw.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[viz] episode {ep}: {T} steps -> {path}")

        # front-camera strip for the contact sheet
        cam = "front_rgb" if "front_rgb" in frames else cams[0]
        ts = np.linspace(0, T - 1, args.sheet_cols).astype(int)
        sheet_rows.append(np.concatenate([frames[cam][t] for t in ts], axis=1))

    sheet = np.concatenate(sheet_rows, axis=0)
    sheet_path = os.path.join(args.out, "contact_sheet.png")
    cv2.imwrite(sheet_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[viz] contact sheet ({len(picks)} episodes x {args.sheet_cols} steps): {sheet_path}")


if __name__ == "__main__":
    main()
