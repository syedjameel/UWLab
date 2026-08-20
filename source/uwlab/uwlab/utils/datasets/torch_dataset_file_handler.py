# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Torch Dataset File Handler
This module provides a dataset file handler that saves data directly in the torch format.
"""

from __future__ import annotations

import os
import torch
from collections.abc import Iterable
from typing import Any

from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase, EpisodeData


def atomic_torch_save(obj, path: str) -> None:
    """Write ``obj`` to ``path`` via a temp file + ``os.replace``, never a direct truncate-in-place
    ``torch.save(obj, path)``.

    A naive in-place ``torch.save`` truncates the destination on open and rewrites it in place;
    sampled concurrently, the file is 0 bytes or a torn partial write for most of the time it
    takes to serialize, so a kill (or crash) landing in that window destroys the whole file. This
    already happened in production. ``os.replace`` is atomic within a filesystem, so the temp file
    MUST live in the same directory as ``path`` -- a temp file under ``/tmp`` would risk a
    cross-device rename (fails, or silently degrades to a non-atomic copy). The temp name includes
    the pid so two concurrent writers of different ``path``s can never collide and a crashed
    process's leftover temp is never picked up later.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class TorchDatasetFileHandler(DatasetFileHandlerBase):
    """
    Dataset file handler that saves data directly in torch format as a torch file.
    """

    def __init__(self):
        self._file_path: str | None = None
        self._episode_data: dict[str, Any] = {}
        self._episode_count: int = 0

    def create(self, file_path: str, env_name: str | None = None):
        """Create a new dataset file."""
        self._file_path = file_path
        self._episode_data = {}
        self._episode_count = 0

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if mode == "r":
            if os.path.exists(file_path):
                self._episode_data = torch.load(file_path)
                self._file_path = file_path
            else:
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
        else:
            self._file_path = file_path
            self._episode_data = {}

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        return []

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
        """
        if episode.is_empty() or not episode.success:
            return

        # Extract only the last entry from this episode using extend_dicts_last_entry logic
        self._extend_dicts_last_entry(self._episode_data, episode.data)

        # Only increment episode count if using default indexing
        if demo_id is None:
            self._episode_count += 1

    def _extend_dicts_last_entry(self, dest: dict[str, Any], src: dict[str, Any], store_data: bool = True) -> None:
        """Extend destination dictionary with last entry from source dictionary."""
        for key in src.keys():
            if isinstance(src[key], dict):
                if key not in dest:
                    dest[key] = {}
                self._extend_dicts_last_entry(dest[key], src[key], store_data)
            else:
                if key not in dest:
                    dest[key] = []
                if store_data:
                    dest[key].extend(src[key][-1:])  # Only take the last entry from src[key]

    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        """Load episode data from the file."""
        # Not applicable for this handler since we store aggregated data
        raise NotImplementedError("Load episode not supported for preprocessed format")

    def flush(self):
        """Flush any pending data to disk."""
        if self._file_path and self._episode_data:
            atomic_torch_save(self._episode_data, self._file_path)

    def close(self):
        """Close the dataset file handler."""
        self.flush()
        self._episode_data = {}
        self._file_path = None

    def add_env_args(self, env_args: dict):
        pass
