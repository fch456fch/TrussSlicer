"""STL/OBJ 网格加载。"""
from __future__ import annotations

from pathlib import Path

import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """加载网格文件。返回 trimesh.Trimesh，已平移到 z>=0 的位置。"""
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"文件 {path} 加载结果不是单一网格")

    # 把模型底部对齐到 z=0
    z_min = mesh.bounds[0, 2]
    if z_min != 0.0:
        mesh.apply_translation([0, 0, -z_min])

    return mesh


def mesh_info(mesh: trimesh.Trimesh) -> dict:
    """返回包围盒、是否水密等基础信息。"""
    bounds = mesh.bounds
    return {
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "size": (bounds[1] - bounds[0]).tolist(),
        "is_watertight": bool(mesh.is_watertight),
        "n_faces": len(mesh.faces),
    }
