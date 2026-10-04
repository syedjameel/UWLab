"""Minimal pure-torch stand-in for the three pytorch3d symbols UWLab's OmniReset
pipeline uses. NOT the real pytorch3d: upstream ships no wheel for
torch 2.11 / cu130 / cp312, and a source build needs nvcc, which this box lacks.

Only `Meshes`, `sample_points_from_meshes` and `sample_farthest_points` are
provided, matching upstream call semantics (see uwlab_tasks .../omnireset/mdp/utils.py).
"""
__version__ = "0.0.0-uwlab-compat"
