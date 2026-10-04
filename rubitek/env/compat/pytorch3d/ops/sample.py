from __future__ import annotations
import torch


def sample_points_from_meshes(meshes, num_samples: int = 10000):
    """Uniform-by-area surface sampling, one batch row per mesh -> (B, num_samples, 3).

    Triangles are drawn with probability proportional to area, then a point is
    drawn uniformly inside the chosen triangle via the standard sqrt barycentric
    map. Matches upstream's default return (points only, no normals).
    """
    verts_list, faces_list = meshes.verts_list(), meshes.faces_list()
    device = meshes.device
    out = torch.empty((len(verts_list), num_samples, 3), device=device, dtype=torch.float32)

    for i, (v, f) in enumerate(zip(verts_list, faces_list)):
        v = v.to(device)
        f = f.to(device)
        if f.numel() == 0 or v.numel() == 0:
            out[i] = 0.0
            continue
        tris = v[f]                                    # (F, 3, 3)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        areas = torch.cross(b - a, c - a, dim=1).norm(dim=1) * 0.5
        total = areas.sum()
        if not torch.isfinite(total) or total <= 0:
            # Degenerate mesh: fall back to sampling vertices so callers still get points.
            idx = torch.randint(0, v.shape[0], (num_samples,), device=device)
            out[i] = v[idx]
            continue
        tri_idx = torch.multinomial(areas / total, num_samples, replacement=True)
        a, b, c = a[tri_idx], b[tri_idx], c[tri_idx]
        u = torch.rand(num_samples, 1, device=device)
        w = torch.rand(num_samples, 1, device=device)
        su = u.sqrt()
        out[i] = (1.0 - su) * a + su * (1.0 - w) * b + su * w * c
    return out


def sample_farthest_points(points: torch.Tensor, lengths=None, K: int = 50,
                           random_start_point: bool = False):
    """Greedy farthest-point sampling -> ((B, K, 3), (B, K) indices).

    Same contract as upstream: iteratively take the point maximising the distance
    to the set already chosen. Upstream starts at index 0 unless
    ``random_start_point``; this does the same so results are reproducible under
    the caller's ``temporary_seed``.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    B, P, D = points.shape
    K = int(min(K, P))
    device = points.device

    if lengths is None:
        lengths = torch.full((B,), P, dtype=torch.long, device=device)
    else:
        lengths = lengths.to(device)

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    if random_start_point:
        start = (torch.rand(B, device=device) * lengths.float()).long().clamp_(max=P - 1)
    else:
        start = torch.zeros(B, dtype=torch.long, device=device)

    arange_p = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
    valid = arange_p < lengths.unsqueeze(1)

    cur = start
    idx[:, 0] = cur
    batch = torch.arange(B, device=device)
    # Running min squared distance from each point to the selected set.
    best = (points - points[batch, cur].unsqueeze(1)).pow(2).sum(-1)
    best = torch.where(valid, best, torch.full_like(best, -1.0))

    for k in range(1, K):
        cur = best.argmax(dim=1)
        idx[:, k] = cur
        d = (points - points[batch, cur].unsqueeze(1)).pow(2).sum(-1)
        best = torch.minimum(best, torch.where(valid, d, torch.full_like(d, -1.0)))
        best = torch.where(valid, best, torch.full_like(best, -1.0))

    return points[batch.unsqueeze(1), idx], idx
