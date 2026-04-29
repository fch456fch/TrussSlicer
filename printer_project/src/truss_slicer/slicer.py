"""平面切片：网格 → 每层 2D 多边形（shapely）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


@dataclass
class Layer:
    index: int
    z: float
    polygons: MultiPolygon  # 该层的 2D 截面（XY 平面）


def slice_mesh(mesh: trimesh.Trimesh, layer_height: float) -> list[Layer]:
    """
    用平面 z = (i + 0.5) * layer_height 切网格，得到每层闭合多边形。
    每个 Z 位于层中心，避免擦边切到面。
    """
    if layer_height <= 0:
        raise ValueError("layer_height 必须 > 0")

    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
    n_layers = int(np.floor((z_max - z_min) / layer_height))
    if n_layers <= 0:
        return []

    layers: list[Layer] = []
    for i in range(n_layers):
        z_cut = z_min + (i + 0.5) * layer_height  # 切片平面在层中心（避免擦边）
        z_top = z_min + (i + 1) * layer_height     # 层顶面高度
        section = mesh.section(plane_origin=[0, 0, z_cut], plane_normal=[0, 0, 1])
        if section is None:
            continue

        planar, _to_3d = section.to_planar()
        polys: list[Polygon] = []
        for p in planar.polygons_full:
            if p.is_valid and p.area > 0:
                polys.append(p)

        if not polys:
            continue

        merged = unary_union(polys)
        if isinstance(merged, Polygon):
            mp = MultiPolygon([merged])
        elif isinstance(merged, MultiPolygon):
            mp = merged
        else:
            continue

        layers.append(Layer(index=i, z=z_top, polygons=mp))

    return layers
