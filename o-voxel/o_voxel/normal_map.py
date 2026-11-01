"""
Tangent-space normal map baking helpers.

A decimated mesh loses fine surface detail. Baking the normals of the original
high-resolution mesh into a tangent-space normal map restores that detail at
render time, so a low face count mesh still looks detailed.

All functions are pure torch and run on any device.
"""
import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Area-weighted vertex normals.

    Args:
        vertices: (V, 3) float tensor
        faces: (F, 3) integer tensor
    Returns:
        (V, 3) unit normals
    """
    faces = faces.long()
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_n = torch.cross(v1 - v0, v2 - v0, dim=-1)  # length = 2 * area
    normals = torch.zeros_like(vertices)
    for k in range(3):
        normals.index_add_(0, faces[:, k], face_n)
    return _normalize(normals)


def face_uv_gradients(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor):
    """
    Per-face position derivatives with respect to the UV coordinates.

    Args:
        vertices: (V, 3) positions
        faces: (F, 3) indices into vertices and uvs
        uvs: (V, 2) texture coordinates
    Returns:
        dp_du, dp_dv: (F, 3) each
    """
    faces = faces.long()
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    t0, t1, t2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    r = (1.0 / det)[:, None]
    dp_du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    dp_dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    return dp_du, dp_dv


def encode_tangent_space(
    detail_normals: torch.Tensor,
    base_normals: torch.Tensor,
    dp_du: torch.Tensor,
    dp_dv: torch.Tensor,
) -> torch.Tensor:
    """
    Express detail normals in the tangent frame of the base (low-poly) surface.

    Uses the glTF convention: +X follows increasing U, +Y points up in the
    texture image (decreasing V, since glTF V runs top to bottom), +Z is the
    base surface normal.

    Args:
        detail_normals: (N, 3) normals of the high-resolution surface
        base_normals: (N, 3) interpolated normals of the low-poly surface
        dp_du, dp_dv: (N, 3) UV gradients of the low-poly face under each sample
    Returns:
        (N, 3) unit tangent-space normals with z >= 0
    """
    n = _normalize(base_normals)
    d = _normalize(detail_normals)
    # Both meshes describe the same surface, so the detail normal must face the same side.
    d = torch.where((d * n).sum(-1, keepdim=True) < 0, -d, d)

    t = dp_du - n * (n * dp_du).sum(-1, keepdim=True)
    t = _normalize(t)
    b = torch.cross(n, t, dim=-1)
    # Handedness: bitangent must point along -dP/dv (image up).
    sign = torch.where((b * -dp_dv).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    b = b * sign

    ts = torch.stack([(d * t).sum(-1), (d * b).sum(-1), (d * n).sum(-1)], dim=-1)
    ts[:, 2] = ts[:, 2].clamp(min=1e-3)
    return _normalize(ts)


def tangent_space_to_rgb(ts: torch.Tensor) -> torch.Tensor:
    """Map unit vectors in [-1, 1] to uint8 RGB."""
    return ((ts * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
