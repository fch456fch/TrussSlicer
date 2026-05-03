"""Stub 路径排序：按边界环顺时针顺序排列打印路径。"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LinearRing, Point

from . import geometry_utils as gu


def _point_on_ring_angle(ring: LinearRing, pt: tuple[float, float], origin: tuple[float, float] = None) -> float:
    """计算点在 ring 上的顺时针角度（以 ring 边界框中心为原点）。"""
    if origin is None:
        bounds = ring.bounds
        origin = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    dx = pt[0] - origin[0]
    dy = pt[1] - origin[1]
    # atan2 返回 (-pi, pi]，转顺时针: -atan2(dy, dx)
    angle = -np.arctan2(dy, dx)
    return angle


def _sort_points_on_ring(
    points: list[tuple[float, float]],
    ring: LinearRing,
) -> list[tuple[float, float]]:
    """按边界环顺时针顺序排序端点。"""
    if not points:
        return []
    bounds = ring.bounds
    origin = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    return sorted(points, key=lambda p: _point_on_ring_angle(ring, p, origin))


def _split_3_stubs(stubs: list[tuple[np.ndarray, str]]) -> list[list[tuple[np.ndarray, str]]]:
    """3 stubs 按 180 度拆分: AC 配对，B 单独。返回 2 个 group。"""
    # stubs: [(coords_2d, direction)]，coords_2d = [internal, boundary]
    # 取 boundary 端点坐标
    boundary_pts = [s[0][1, :2] for s in stubs]  # (3, 2)

    # 找 AC（180度的两条），B 是中间那条
    # 简化：用 boundary 端点间的夹角判断
    angles = []
    for i, (bp, _) in enumerate(stubs):
        angles.append((i, bp))

    # 计算两两夹角，找最大（180度）
    best_pair = None
    max_dist = -1.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(angles[i][1] - angles[j][1])
            if d > max_dist:
                max_dist = d
                best_pair = (i, j)

    a_idx, c_idx = best_pair
    b_idx = 3 - a_idx - c_idx  # 剩下那个

    group_ac = [stubs[a_idx], stubs[c_idx]]
    group_b = [stubs[b_idx]]
    return [group_ac, group_b]


def _get_cw_start_boundary(stub: tuple[np.ndarray, str], ring: LinearRing) -> float:
    """获取 stub 在边界上的端点，按 cw 角度排序的位置（作为组间排序 key）。"""
    bp = stub[0][1, :2]
    return _point_on_ring_angle(ring, tuple(bp))


def _get_cw_start_chain(chain: list[tuple[np.ndarray, str]], ring: LinearRing) -> float:
    """获取 chain 的逆时针起点（cw 排序 key）。"""
    if len(chain) == 1:
        # 1-stub：boundary 端点就是起点
        return _get_cw_start_boundary(chain[0], ring)
    elif len(chain) == 2:
        # 2-stub：取逆时针方向先到达的端点
        angles = [_get_cw_start_boundary(s, ring) for s in chain]
        return min(angles)  # cw 角度最小 = 逆时针最靠前
    return 0.0


def _build_chain_order(
    stubs: list[tuple[np.ndarray, str]],
    boundaries: list[LinearRing],
    edge_lines: list[np.ndarray],
) -> tuple[list[np.ndarray], list]:
    """按边界环 cw 顺序排列所有 stub chains 和 edge_lines。

    Returns:
        ordered_paths: 排序后的路径列表（2D coords）
        visit_markers: 标记列表（用于调试）
    """
    if not stubs and not edge_lines:
        return [], []

    # 第一步：按 internal 节点分组
    groups: dict[tuple, list] = {}
    tol = 0.01
    for coords, direction in stubs:
        key = (round(coords[0, 0] / tol), round(coords[0, 1] / tol))
        groups.setdefault(key, []).append((coords, direction))

    # 第二步：每组内拆分 3-stub → 2 groups
    all_chains: list[tuple[list, LinearRing]] = []  # (chain stubs, boundary ring)

    for group in groups.values():
        # 该组所有 stubs 的 boundary 端点分布在哪些 ring 上
        # 简化：取第一个 stub 的 boundary 端点找 ring
        if not group:
            continue

        # 找每个 stub 的 boundary 端点属于哪个 ring
        for coords, direction in group:
            bp = tuple(coords[1, :2])
            ring = _find_closest_ring(bp, boundaries)

        # 边界端点按 cw 排序
        sorted_group = sorted(group, key=lambda s: _get_cw_start_boundary(s, ring))

        if len(sorted_group) == 1:
            all_chains.append((sorted_group, ring))
        elif len(sorted_group) == 2:
            all_chains.append((sorted_group, ring))
        elif len(sorted_group) == 3:
            # 拆分 AC+B
            sub_groups = _split_3_stubs(sorted_group)
            for sg in sub_groups:
                all_chains.append((sg, ring))

    # 第三步：edge_lines 也参与排序（每条属于其起点所在的 ring）
    # edge_lines: list[np.ndarray], shape (n, 2)
    edge_info = []  # (edge_line, start_ring, start_angle)
    for e2d in edge_lines:
        if len(e2d) < 2:
            continue
        start_pt = tuple(e2d[0, :2])
        start_ring = _find_closest_ring(start_pt, boundaries)
        angle = _point_on_ring_angle(start_ring, start_pt)
        edge_info.append((e2d, start_ring, angle))

    # 第四步：合并所有端点（stubs chains + edge_lines），按 ring 和 cw 排序
    # 每个 chain 贡献其 "逆时针起点" 端点
    ordered = []  # (sort_key, item_type, item_data, ring)
    # item_type: "stub_chain" | "edge_line"

    for chain_stubs, ring in all_chains:
        if not chain_stubs:
            continue
        key = _get_cw_start_chain(chain_stubs, ring)
        ordered.append((key, "stub_chain", chain_stubs, ring))

    for e2d, ring, angle in edge_info:
        ordered.append((angle, "edge_line", e2d, ring))

    # 按 ring 分组，同 ring 内按 cw 排序
    by_ring: dict[int, list] = {}
    ring_ids: dict[int, int] = {}  # ring object -> id for dict key
    for key, item_type, item_data, ring_obj in ordered:
        rid = id(ring_obj)
        if rid not in by_ring:
            by_ring[rid] = []
            ring_ids[rid] = ring_obj
        by_ring[rid].append((key, item_type, item_data))

    # 每 ring 内按 cw 排序
    for rid in by_ring:
        by_ring[rid].sort(key=lambda x: x[0])

    # 第五步：遍历，收集 visit 标记
    # edge_lines 的 visit 用 set 记录
    visited_edges: set[int] = set()
    result_paths = []
    visit_markers = []

    for rid, items in by_ring.items():
        for sort_key, item_type, item_data in items:
            if item_type == "stub_chain":
                result_paths.append(("stub_chain", item_data))
                visit_markers.append(("stub", sort_key))
            elif item_type == "edge_line":
                # 检查是否已 visit（通过边的端点坐标匹配）
                e2d = item_data
                edge_id = hash(tuple(map(tuple, e2d.tolist())))
                if edge_id in visited_edges:
                    visit_markers.append(("edge_skip", sort_key))
                    continue
                result_paths.append(("edge_line", e2d))
                visited_edges.add(edge_id)
                visit_markers.append(("edge", sort_key))

    return result_paths, visit_markers


def _find_closest_ring(
    pt: tuple[float, float],
    rings: list[LinearRing],
) -> LinearRing:
    """找距离点最近的 ring。"""
    p = Point(pt)
    best_ring = rings[0] if rings else None
    best_dist = float("inf")
    for ring in rings:
        d = ring.distance(p)
        if d < best_dist:
            best_dist = d
            best_ring = ring
    return best_ring


def plan_stub_paths_sorted(
    stubs: list[tuple[np.ndarray, str]],
    boundaries: list[LinearRing],
    edge_lines: list[np.ndarray],
    rivet_len: float = 1.0,
) -> list[np.ndarray]:
    """按边界环 cw 顺序排列 stub chains 和 edge_lines，返回排序后的路径列表。"""
    ordered, _ = _build_chain_order(stubs, boundaries, edge_lines)

    result: list[np.ndarray] = []

    for item_type, item_data in ordered:
        if item_type == "stub_chain":
            # 合成连续路径：一笔打印
            path = _merge_stub_chain(item_data)
            path = _add_stub_rivets_2d(path, boundaries, rivet_len)
            result.append(path)
        elif item_type == "edge_line":
            e2d = item_data
            path = _add_stub_rivets_2d(e2d, boundaries, rivet_len)
            result.append(path)

    return result


def _merge_stub_chain(stub_chain: list[tuple[np.ndarray, str]]) -> np.ndarray:
    """将同 internal 节点的 stubs 合并成一条连续路径。

    coords 格式: [internal, boundary]
    1 stub → boundary → internal
    2 stubs → boundary A → internal → boundary B
    """
    if len(stub_chain) == 1:
        coords, _ = stub_chain[0]
        # 翻转: [boundary, internal]
        return coords[::-1]
    elif len(stub_chain) == 2:
        c0, _ = stub_chain[0]
        c1, _ = stub_chain[1]
        # c0: [internal, boundary A] → 翻转 → [boundary A, internal]
        # c1: [internal, boundary B] → 保持 → [internal, boundary B]
        return np.vstack([c0[::-1], c1[1:]])
    return np.array([])


def _add_stub_rivets_2d(
    coords: np.ndarray,
    boundaries: list[LinearRing],
    rivet_len: float,
) -> np.ndarray:
    """在路径首尾的 boundary 端加铆线（2D 版本）。"""
    from shapely.geometry import Point
    tol = rivet_len * 0.5
    path = coords.copy()

    def _find_ring(x, y):
        pt = Point(x, y)
        best_ri, best_d = 0, float("inf")
        for ri, ring in enumerate(boundaries):
            d = ring.distance(pt)
            if d < best_d:
                best_d = d
                best_ri = ri
        return best_ri, best_d

    # 首端
    ri, d = _find_ring(path[0, 0], path[0, 1])
    if d < tol:
        ring = boundaries[ri]
        t = ring.project(Point(path[0, 0], path[0, 1]))
        t_start = (t - rivet_len) % ring.length
        arc = gu.shortest_arc(ring, t_start, t)
        if arc is not None and len(arc) >= 2:
            path = np.vstack([arc, path])

    # 尾端
    ri, d = _find_ring(path[-1, 0], path[-1, 1])
    if d < tol:
        ring = boundaries[ri]
        t = ring.project(Point(path[-1, 0], path[-1, 1]))
        t_end = (t + rivet_len) % ring.length
        arc = gu.shortest_arc(ring, t, t_end)
        if arc is not None and len(arc) >= 2:
            path = np.vstack([path, arc])

    return path
