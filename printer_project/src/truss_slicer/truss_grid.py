"""
桁架网格生成（移植自 rhino-gh/桁架网络.py + gh_path_tools.py）。

生成带 (i+j+k)%2 奇偶交替 Z 高度的 3D 网格线。
边界节点不再压平——端点 Z 继承最近内部节点的 Z。
线段按两端 Z 分为 bottom / top / mixed，mixed 在低位内部节点处切开。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from .snake_planner import GridLine


@dataclass
class TrussGridResult:
    bottom_lines: list[GridLine] = field(default_factory=list)
    top_lines: list[GridLine] = field(default_factory=list)


def _is_inside(polygon: Polygon | MultiPolygon, x: float, y: float) -> bool:
    return polygon.contains(Point(x, y))


def _is_boundary_node(polygon: Polygon | MultiPolygon, x: float, y: float, spacing: float) -> bool:
    for nx, ny in [(x + spacing, y), (x - spacing, y), (x, y + spacing), (x, y - spacing)]:
        if not _is_inside(polygon, nx, ny):
            return True
    return False


def _node_z(i: int, j: int, k: int, z_low: float, z_high: float) -> float:
    return z_low if (i + j + k) % 2 == 0 else z_high


def _classify_and_split(
    lines: list[GridLine], z_low: float, z_high: float,
) -> TrussGridResult:
    """
    按两端 Z 分类：同低→bottom, 同高→top, 异侧→在低位内部节点切开。
    """
    result = TrussGridResult()
    z_mid = (z_low + z_high) / 2

    for gl in lines:
        pts = gl.coords
        if len(pts) < 2:
            continue
        z_start = pts[0, 2]
        z_end = pts[-1, 2]
        start_is_low = z_start < z_mid
        end_is_low = z_end < z_mid

        if start_is_low and end_is_low:
            result.bottom_lines.append(gl)
        elif not start_is_low and not end_is_low:
            result.top_lines.append(gl)
        else:
            _split_mixed(gl, z_low, z_mid, result)

    return result


def _split_mixed(gl: GridLine, z_low: float, z_mid: float, result: TrussGridResult) -> None:
    """异侧线在低位内部节点处切开，分别归入 bottom/top。"""
    pts = gl.coords
    # 找中间的低位点（排除首尾端点）
    low_indices = [i for i in range(1, len(pts) - 1) if pts[i, 2] < z_mid]

    if not low_indices:
        # 没有内部低位点可切，整条归入端点 Z 较低的一侧
        if pts[0, 2] < z_mid:
            result.bottom_lines.append(gl)
        else:
            result.top_lines.append(gl)
        return

    # 选最靠近中间的低位点
    mid_idx = len(pts) // 2
    split_idx = min(low_indices, key=lambda i: abs(i - mid_idx))

    # 低端半段 → bottom, 高端半段 → top
    part_a = pts[: split_idx + 1]
    part_b = pts[split_idx:]

    if len(part_a) >= 2:
        a_start_low = part_a[0, 2] < z_mid
        if a_start_low:
            result.bottom_lines.append(GridLine(coords=part_a, direction=gl.direction, dead_end=True))
        else:
            result.top_lines.append(GridLine(coords=part_a, direction=gl.direction, dead_end=True))

    if len(part_b) >= 2:
        b_end_low = part_b[-1, 2] < z_mid
        if b_end_low:
            result.bottom_lines.append(GridLine(coords=part_b, direction=gl.direction, dead_start=True))
        else:
            result.top_lines.append(GridLine(coords=part_b, direction=gl.direction, dead_start=True))


def generate_truss_grid(
    infill_polygon: Polygon | MultiPolygon,
    spacing: float,
    z_low: float,
    z_high: float,
    cell_index: int = 0,
    origin: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
) -> TrussGridResult:
    if infill_polygon is None or infill_polygon.is_empty:
        return TrussGridResult()

    if isinstance(infill_polygon, Polygon):
        infill_polygon = MultiPolygon([infill_polygon])

    from shapely import affinity
    from shapely.geometry import LineString

    bx = infill_polygon.bounds
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    rotated_poly = affinity.rotate(infill_polygon, -angle, origin=(cx, cy))

    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rotate_back_3d(pts: list[list[float]]) -> list[list[float]]:
        out = []
        for x, y, z in pts:
            dx, dy = x - cx, y - cy
            out.append([dx * cos_a - dy * sin_a + cx, dx * sin_a + dy * cos_a + cy, z])
        return out

    bounds = rotated_poly.bounds
    minx, miny, maxx, maxy = bounds
    ox, oy = cx, cy

    start_i = int(math.floor((minx - ox) / spacing)) - 1
    end_i = int(math.ceil((maxx - ox) / spacing)) + 1
    start_j = int(math.floor((miny - oy) / spacing)) - 1
    end_j = int(math.ceil((maxy - oy) / spacing)) + 1

    raw_lines: list[GridLine] = []
    k = cell_index

    # X 方向线（y 固定，j 固定）
    for j in range(start_j, end_j + 1):
        y = oy + j * spacing
        scan = LineString([(minx - spacing, y), (maxx + spacing, y)])
        inter = rotated_poly.intersection(scan)
        for seg in _extract_segments(inter):
            coords_2d = np.asarray(seg.coords)
            if len(coords_2d) < 2:
                continue
            if coords_2d[0, 0] > coords_2d[-1, 0]:
                coords_2d = coords_2d[::-1]

            pts_3d = []
            p_s, p_e = coords_2d[0], coords_2d[-1]
            i_min = int(math.floor((p_s[0] - ox) / spacing))
            i_max = int(math.ceil((p_e[0] - ox) / spacing))

            internal_pts = []
            for i in range(i_min, i_max + 1):
                x = ox + i * spacing
                if p_s[0] + 1e-4 < x < p_e[0] - 1e-4:
                    z = _node_z(i, j, k, z_low, z_high)
                    internal_pts.append([x, y, z])

            # n_spacing = 经过的 spacing 数量（向下取整）
            # n_spacing <= 2：跳过，避免 low-high 跨接问题
            n_spacing = len(internal_pts)
            if n_spacing <= 2:
                continue

            # 端点 Z 继承最近内部节点
            z_first = internal_pts[0][2]
            z_last = internal_pts[-1][2]

            pts_3d.append([p_s[0], p_s[1], z_first])
            pts_3d.extend(internal_pts)
            pts_3d.append([p_e[0], p_e[1], z_last])

            if len(pts_3d) >= 2:
                pts_3d = rotate_back_3d(pts_3d)
                raw_lines.append(GridLine(coords=np.array(pts_3d), direction="X"))

    # Y 方向线（x 固定，i 固定）
    for i in range(start_i, end_i + 1):
        x = ox + i * spacing
        scan = LineString([(x, miny - spacing), (x, maxy + spacing)])
        inter = rotated_poly.intersection(scan)
        for seg in _extract_segments(inter):
            coords_2d = np.asarray(seg.coords)
            if len(coords_2d) < 2:
                continue
            if coords_2d[0, 1] > coords_2d[-1, 1]:
                coords_2d = coords_2d[::-1]

            pts_3d = []
            p_s, p_e = coords_2d[0], coords_2d[-1]
            j_min = int(math.floor((p_s[1] - oy) / spacing))
            j_max = int(math.ceil((p_e[1] - oy) / spacing))

            internal_pts = []
            for j in range(j_min, j_max + 1):
                y = oy + j * spacing
                if p_s[1] + 1e-4 < y < p_e[1] - 1e-4:
                    z = _node_z(i, j, k, z_low, z_high)
                    internal_pts.append([x, y, z])

            # n_spacing = 经过的 spacing 数量（向下取整）
            # n_spacing <= 2：跳过，避免 low-high 跨接问题
            n_spacing = len(internal_pts)
            if n_spacing <= 2:
                continue

            # 端点 Z 继承最近内部节点
            z_first = internal_pts[0][2]
            z_last = internal_pts[-1][2]

            pts_3d.append([p_s[0], p_s[1], z_first])
            pts_3d.extend(internal_pts)
            pts_3d.append([p_e[0], p_e[1], z_last])

            if len(pts_3d) >= 2:
                pts_3d = rotate_back_3d(pts_3d)
                raw_lines.append(GridLine(coords=np.array(pts_3d), direction="Y"))

    return _classify_and_split(raw_lines, z_low, z_high)


def _extract_segments(geom) -> list:
    from shapely.geometry import LineString
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []
