"""共用几何工具函数。"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LinearRing, LineString, Point
from shapely.ops import substring


def project_point_on_ring(ring: LinearRing, pt: Point | tuple) -> float:
    """返回点在闭合环上的投影距离参数 (0 ~ ring.length)。"""
    if not isinstance(pt, Point):
        pt = Point(pt[0], pt[1])
    return ring.project(pt)


def shortest_arc(ring: LinearRing, d1: float, d2: float) -> np.ndarray | None:
    """
    在闭合环上取 d1→d2 的最短弧段，返回 (N,2) 坐标数组。
    等价于 rhino-gh 的 get_boundary_segment（双向取最短）。
    """
    length = ring.length
    if length < 1e-9:
        return None
    if abs(d1 - d2) < 1e-6:
        return None

    # 正向弧 d1→d2
    if d1 <= d2:
        fwd = substring(ring, d1, d2)
    else:
        # 跨接缝：d1→length + 0→d2
        seg_a = substring(ring, d1, length)
        seg_b = substring(ring, 0, d2)
        fwd = _join_linestrings(seg_a, seg_b)

    # 反向弧 d2→d1（即 d1←d2 方向，但我们取 d2→d1 再反转）
    if d2 <= d1:
        bwd = substring(ring, d2, d1)
    else:
        seg_a = substring(ring, d2, length)
        seg_b = substring(ring, 0, d1)
        bwd = _join_linestrings(seg_a, seg_b)

    fwd_len = fwd.length if fwd and not fwd.is_empty else float("inf")
    bwd_len = bwd.length if bwd and not bwd.is_empty else float("inf")

    if fwd_len == float("inf") and bwd_len == float("inf"):
        return None

    if fwd_len <= bwd_len:
        chosen = fwd
    else:
        # bwd 是 d2→d1 方向，需要反转成 d1→d2
        coords = list(bwd.coords)
        coords.reverse()
        chosen = LineString(coords)

    coords = np.asarray(chosen.coords)
    return coords if len(coords) >= 2 else None


def _join_linestrings(a: LineString, b: LineString) -> LineString:
    """拼接两段 LineString。"""
    if a is None or a.is_empty:
        return b
    if b is None or b.is_empty:
        return a
    coords = list(a.coords) + list(b.coords)[1:]
    return LineString(coords)
