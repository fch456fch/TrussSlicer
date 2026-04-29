"""
蛇形路径规划器（移植自 rhino-gh/路径规划.py + gh_path_tools.py）。

通用 2D 拓扑蛇形，同时服务于实心层填充、锚定层网格、桁架本体。
桁架本体的端点 Z 属性对本模块透明——输入线段带 Z，输出折线保留 Z。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.ops import substring

from . import geometry_utils as gu

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GridLine:
    """一根网格线。coords 是 (N, 2) 或 (N, 3) 的 numpy 数组。"""
    coords: np.ndarray
    direction: str
    dead_start: bool = False
    dead_end: bool = False

    @property
    def start(self) -> np.ndarray:
        return self.coords[0]

    @property
    def end(self) -> np.ndarray:
        return self.coords[-1]


# ---------------------------------------------------------------------------
# 网格线生成（平面版，用于实心层 / 锚定层）
# ---------------------------------------------------------------------------

def generate_grid_lines(
    infill_region: MultiPolygon | Polygon,
    spacing: float,
    angle: float = 0.0,
) -> list[GridLine]:
    """
    在 infill_region 内生成正交网格线（X 向 + Y 向）。
    angle 为旋转角度（度），默认 0 = 轴对齐，45 = 菱形网格。
    实现方式：将区域反向旋转 → 生成轴对齐网格 → 将线段正向旋转回去。
    """
    if infill_region is None or infill_region.is_empty:
        return []

    if isinstance(infill_region, Polygon):
        infill_region = MultiPolygon([infill_region])

    from shapely import affinity

    rad = np.radians(angle)
    # 旋转中心 = bbox 中心
    bx = infill_region.bounds
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2

    rotated_region = affinity.rotate(infill_region, -angle, origin=(cx, cy))

    lines: list[GridLine] = []
    minx, miny, maxx, maxy = rotated_region.bounds

    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rotate_back(coords_2d: np.ndarray) -> np.ndarray:
        dx = coords_2d[:, 0] - cx
        dy = coords_2d[:, 1] - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        return np.column_stack([rx, ry])

    # 从中心向两侧展开，保证对称
    n_half = int(np.ceil((maxy - cy) / spacing))
    for j in range(-n_half, n_half + 1):
        y = cy + j * spacing
        if y <= miny or y >= maxy:
            continue
        scan = LineString([(minx - spacing, y), (maxx + spacing, y)])
        inter = rotated_region.intersection(scan)
        for seg in _extract_linestrings(inter):
            coords = np.asarray(seg.coords)
            if len(coords) >= 2:
                coords = rotate_back(coords)
                lines.append(GridLine(coords=coords, direction="X"))

    n_half = int(np.ceil((maxx - cx) / spacing))
    for i in range(-n_half, n_half + 1):
        x = cx + i * spacing
        if x <= minx or x >= maxx:
            continue
        scan = LineString([(x, miny - spacing), (x, maxy + spacing)])
        inter = rotated_region.intersection(scan)
        for seg in _extract_linestrings(inter):
            coords = np.asarray(seg.coords)
            if len(coords) >= 2:
                coords = rotate_back(coords)
                lines.append(GridLine(coords=coords, direction="Y"))

    return lines


def generate_parallel_lines(
    infill_region: MultiPolygon | Polygon,
    spacing: float,
    angle: float = 0.0,
) -> list[GridLine]:
    """
    生成单方向平行线（用于实心层）。
    angle=0 → X 方向水平线，angle=90 → Y 方向竖直线。
    """
    if infill_region is None or infill_region.is_empty:
        return []

    if isinstance(infill_region, Polygon):
        infill_region = MultiPolygon([infill_region])

    from shapely import affinity

    rad = np.radians(angle)
    bx = infill_region.bounds
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    rotated_region = affinity.rotate(infill_region, -angle, origin=(cx, cy))

    lines: list[GridLine] = []
    minx, miny, maxx, maxy = rotated_region.bounds
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    def rotate_back(coords_2d: np.ndarray) -> np.ndarray:
        dx = coords_2d[:, 0] - cx
        dy = coords_2d[:, 1] - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        return np.column_stack([rx, ry])

    n_half = int(np.ceil((maxy - cy) / spacing))
    for j in range(-n_half, n_half + 1):
        y = cy + j * spacing
        if y <= miny or y >= maxy:
            continue
        scan = LineString([(minx - spacing, y), (maxx + spacing, y)])
        inter = rotated_region.intersection(scan)
        for seg in _extract_linestrings(inter):
            coords = np.asarray(seg.coords)
            if len(coords) >= 2:
                coords = rotate_back(coords)
                lines.append(GridLine(coords=coords, direction="X"))

    return lines


def _extract_linestrings(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            if isinstance(g, LineString) and not g.is_empty:
                out.append(g)
        return out
    return []


# ---------------------------------------------------------------------------
# 蛇形路径规划（移植自 rhino-gh）
# ---------------------------------------------------------------------------

def _get_direction(coords: np.ndarray) -> str:
    dx = abs(coords[-1, 0] - coords[0, 0])
    dy = abs(coords[-1, 1] - coords[0, 1])
    return "X" if dx > dy else "Y"


def _find_nearest_ring(boundaries: list[LinearRing], x: float, y: float) -> int:
    pt = Point(x, y)
    best_idx, best_dist = 0, float("inf")
    for i, ring in enumerate(boundaries):
        d = ring.distance(pt)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _build_boundary_data(boundaries: list[LinearRing], grid_lines: list[GridLine]):
    """
    将网格线端点投影到最近的边界环上，按环分组、按参数排序。
    dead_start/dead_end 的端点不注册。
    返回: line_data, ring_endpoints (per-ring sorted lists), lookup, visited
    """
    line_data = []
    n_rings = len(boundaries)
    ring_buckets: list[list[dict]] = [[] for _ in range(n_rings)]

    for i, gl in enumerate(grid_lines):
        line_data.append({"coords": gl.coords, "dir": gl.direction,
                          "dead_start": gl.dead_start, "dead_end": gl.dead_end})

        if not gl.dead_start:
            ri = _find_nearest_ring(boundaries, gl.start[0], gl.start[1])
            t_s = boundaries[ri].project(Point(gl.start[0], gl.start[1]))
            ring_buckets[ri].append({"t": t_s, "line_idx": i, "is_start": True, "ring_idx": ri})

        if not gl.dead_end:
            ri = _find_nearest_ring(boundaries, gl.end[0], gl.end[1])
            t_e = boundaries[ri].project(Point(gl.end[0], gl.end[1]))
            ring_buckets[ri].append({"t": t_e, "line_idx": i, "is_start": False, "ring_idx": ri})

    for bucket in ring_buckets:
        bucket.sort(key=lambda x: x["t"])

    lookup = {}
    for ri, bucket in enumerate(ring_buckets):
        for local_idx, item in enumerate(bucket):
            lookup[(item["line_idx"], item["is_start"])] = (ri, local_idx)

    visited = [False] * len(grid_lines)
    return line_data, ring_buckets, lookup, visited


def _find_best_neighbor(
    ring_idx: int,
    local_idx: int,
    curr_dir: str,
    ring_buckets: list[list[dict]],
    line_data: list,
    visited: list,
    lookup: dict,
    boundaries: list[LinearRing],
    depth: int = 1,
    max_arc: float = float("inf"),
):
    """DFS 深度预判寻路。在同一个环内 ±1 找邻居。"""
    bucket = ring_buckets[ring_idx]
    count = len(bucket)
    if count == 0:
        return None, 0

    bnd_len = boundaries[ring_idx].length
    t_curr = bucket[local_idx]["t"]
    indices = [(local_idx + 1) % count, (local_idx - 1) % count]
    candidates = []

    for idx in indices:
        l_idx = bucket[idx]["line_idx"]
        if not visited[l_idx]:
            t_cand = bucket[idx]["t"]
            arc = abs(t_cand - t_curr)
            arc = min(arc, bnd_len - arc)
            if arc > max_arc:
                continue
            is_diff = line_data[l_idx]["dir"] != curr_dir
            candidates.append((ring_idx, idx, l_idx, is_diff))

    if not candidates:
        return None, 0

    best_cand = None
    max_score = -1

    for ri, idx, l_idx, is_diff in candidates:
        base_score = 2 if is_diff else 1

        if depth > 0:
            visited[l_idx] = True
            try:
                next_dir = line_data[l_idx]["dir"]
                enter_is_start = bucket[idx]["is_start"]
                exit_key = (l_idx, not enter_is_start)
                if exit_key in lookup:
                    next_ri, next_local = lookup[exit_key]
                    _, future_score = _find_best_neighbor(
                        next_ri, next_local, next_dir, ring_buckets,
                        line_data, visited, lookup, boundaries, depth - 1,
                        max_arc,
                    )
                    total_score = base_score * 10 + future_score
                else:
                    total_score = base_score
            finally:
                visited[l_idx] = False
        else:
            total_score = base_score

        if total_score > max_score:
            max_score = total_score
            best_cand = (ri, idx, l_idx, is_diff)

    return best_cand, max_score


def _add_rivets(path: np.ndarray, boundaries: list[LinearRing], rivet_len: float, arc_z: float | None) -> np.ndarray:
    """在路径首尾各加一小段沿边界的铆线。仅对靠近边界的端点生效。"""
    ndim = path.shape[1]
    tol = rivet_len * 0.5

    head_pt = Point(path[0, 0], path[0, 1])
    ri_head = _find_nearest_ring(boundaries, path[0, 0], path[0, 1])
    bnd = boundaries[ri_head]
    if bnd.distance(head_pt) < tol:
        t_head = bnd.project(head_pt)
        t_rivet_start = (t_head - rivet_len) % bnd.length
        head_arc = gu.shortest_arc(bnd, t_rivet_start, t_head)
        if head_arc is not None and len(head_arc) >= 2:
            if ndim == 3:
                z_val = arc_z if arc_z is not None else float(path[0, 2])
                zs = np.full((len(head_arc), 1), z_val)
                head_arc = np.hstack([head_arc, zs])
            path = np.vstack([head_arc, path])

    tail_pt = Point(path[-1, 0], path[-1, 1])
    ri_tail = _find_nearest_ring(boundaries, path[-1, 0], path[-1, 1])
    bnd = boundaries[ri_tail]
    if bnd.distance(tail_pt) < tol:
        t_tail = bnd.project(tail_pt)
        t_rivet_end = (t_tail + rivet_len) % bnd.length
        tail_arc = gu.shortest_arc(bnd, t_tail, t_rivet_end)
        if tail_arc is not None and len(tail_arc) >= 2:
            if ndim == 3:
                z_val = arc_z if arc_z is not None else float(path[-1, 2])
                zs = np.full((len(tail_arc), 1), z_val)
                tail_arc = np.hstack([tail_arc, zs])
            path = np.vstack([path, tail_arc])

    return path


def plan_snake_paths(
    grid_lines: list[GridLine],
    boundary: LinearRing | list[LinearRing],
    start_depth: int = 3,
    step_depth: int = 2,
    arc_z: float | None = None,
    max_arc_len: float = 20.0,
    rivet_len: float = 1.0,
) -> list[np.ndarray]:
    """PLACEHOLDER_CONTINUE"""
    if not grid_lines:
        return []

    if isinstance(boundary, LinearRing):
        boundaries = [boundary]
    else:
        boundaries = boundary

    line_data, ring_buckets, lookup, visited = _build_boundary_data(boundaries, grid_lines)
    max_arc = max_arc_len
    result_paths: list[np.ndarray] = []

    while True:
        best_start_idx = -1
        best_is_forward = True
        max_score = -1

        for idx in range(len(grid_lines)):
            if visited[idx]:
                continue
            curr_dir = line_data[idx]["dir"]

            visited[idx] = True
            try:
                score_fwd = 0
                score_bwd = 0
                if (idx, False) in lookup:
                    ri_fwd, local_fwd = lookup[(idx, False)]
                    _, score_fwd = _find_best_neighbor(
                        ri_fwd, local_fwd, curr_dir, ring_buckets,
                        line_data, visited, lookup, boundaries,
                        depth=start_depth, max_arc=max_arc,
                    )
                if (idx, True) in lookup:
                    ri_bwd, local_bwd = lookup[(idx, True)]
                    _, score_bwd = _find_best_neighbor(
                        ri_bwd, local_bwd, curr_dir, ring_buckets,
                        line_data, visited, lookup, boundaries,
                        depth=start_depth, max_arc=max_arc,
                    )
            finally:
                visited[idx] = False

            if score_fwd > max_score:
                max_score = score_fwd
                best_start_idx = idx
                best_is_forward = True
            if score_bwd > max_score:
                max_score = score_bwd
                best_start_idx = idx
                best_is_forward = False

        if best_start_idx == -1:
            break

        segments: list[np.ndarray] = []
        c_idx = best_start_idx
        visited[c_idx] = True
        curr_dir = line_data[c_idx]["dir"]
        coords = line_data[c_idx]["coords"]

        if best_is_forward:
            segments.append(coords)
            if line_data[c_idx]["dead_end"] or (c_idx, False) not in lookup:
                exit_loc = None
            else:
                exit_loc = lookup[(c_idx, False)]
        else:
            segments.append(coords[::-1])
            if line_data[c_idx]["dead_start"] or (c_idx, True) not in lookup:
                exit_loc = None
            else:
                exit_loc = lookup[(c_idx, True)]

        while exit_loc is not None:
            exit_ri, exit_local = exit_loc
            next_cand, _ = _find_best_neighbor(
                exit_ri, exit_local, curr_dir, ring_buckets,
                line_data, visited, lookup, boundaries,
                depth=step_depth, max_arc=max_arc,
            )
            if not next_cand:
                break

            _, target_local, next_l_idx, _ = next_cand
            visited[next_l_idx] = True

            t_start = ring_buckets[exit_ri][exit_local]["t"]
            t_end = ring_buckets[exit_ri][target_local]["t"]
            arc = gu.shortest_arc(boundaries[exit_ri], t_start, t_end)
            if arc is not None and len(arc) >= 2:
                segments.append(arc)

            next_coords = line_data[next_l_idx]["coords"]
            enter_is_start = ring_buckets[exit_ri][target_local]["is_start"]
            if enter_is_start:
                segments.append(next_coords)
                if line_data[next_l_idx]["dead_end"] or (next_l_idx, False) not in lookup:
                    exit_loc = None
                else:
                    exit_loc = lookup[(next_l_idx, False)]
            else:
                segments.append(next_coords[::-1])
                if line_data[next_l_idx]["dead_start"] or (next_l_idx, True) not in lookup:
                    exit_loc = None
                else:
                    exit_loc = lookup[(next_l_idx, True)]

            curr_dir = line_data[next_l_idx]["dir"]

        if segments:
            ndim = max(s.shape[1] for s in segments)
            unified: list[np.ndarray] = []
            last_z = arc_z if arc_z is not None else 0.0
            for seg in segments:
                if seg.shape[1] < ndim:
                    z_val = arc_z if arc_z is not None else last_z
                    zs = np.full((len(seg), 1), z_val)
                    seg = np.hstack([seg, zs])
                if len(seg) > 0:
                    last_z = float(seg[-1, 2]) if ndim >= 3 else 0.0
                unified.append(seg)

            all_pts = [unified[0]]
            for seg in unified[1:]:
                if len(seg) > 0:
                    if np.allclose(all_pts[-1][-1][:2], seg[0][:2], atol=0.01):
                        all_pts.append(seg[1:])
                    else:
                        all_pts.append(seg)
            non_empty = [a for a in all_pts if len(a) > 0]
            if non_empty:
                path = np.vstack(non_empty)
                if rivet_len > 0:
                    path = _add_rivets(path, boundaries, rivet_len, arc_z)
                result_paths.append(path)

    return result_paths
