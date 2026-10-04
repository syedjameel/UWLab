from __future__ import annotations
import torch


class Meshes:
    """Heterogeneous batch of triangle meshes.

    Upstream Meshes does far more; this keeps exactly what the sampler needs -
    per-mesh vertex and face tensors, kept as lists because meshes differ in size.
    """

    def __init__(self, verts: list[torch.Tensor], faces: list[torch.Tensor]):
        if len(verts) != len(faces):
            raise ValueError(f"verts/faces length mismatch: {len(verts)} vs {len(faces)}")
        self.verts_list_ = [v.as_subclass(torch.Tensor).float() for v in verts]
        self.faces_list_ = [f.as_subclass(torch.Tensor).long() for f in faces]

    def __len__(self):
        return len(self.verts_list_)

    def verts_list(self):
        return self.verts_list_

    def faces_list(self):
        return self.faces_list_

    @property
    def device(self):
        return self.verts_list_[0].device if self.verts_list_ else torch.device("cpu")
