"""外壳 offset，分离 wall / infill 区域。"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


@dataclass
class LayerRegions:
    wall_centerlines: list[MultiPolygon]  # 从外到内每圈外壳中心线
    infill_region: MultiPolygon | None  # 最内圈包围的填充区域


def build_layer_regions(
    polygon: MultiPolygon,
    wall_count: int,
    line_width: float,
    infill_overlap: float = 0.15,
) -> LayerRegions:
    """
    对每层多边形做 N 圈内缩，得到外壳中心线和填充区域。

    第一圈中心线 = 原轮廓向内偏移 line_width/2，
    后续每圈再偏移 line_width。
    填充区域 = 最后一圈中心线再向内偏移 line_width * (1 - infill_overlap)。
    wall_count=0 时不生成外壳，整个区域都是 infill。
    """
    centerlines: list[MultiPolygon] = []
    current = polygon
    cumulative = 0.0

    for i in range(wall_count):
        d = line_width / 2 if i == 0 else line_width
        cumulative += d
        center = polygon.buffer(-cumulative)
        if center.is_empty:
            break
        centerlines.append(_to_multi(center))
        infill_offset = line_width * (1.0 - infill_overlap)
        inner = center.buffer(-infill_offset)
        if inner.is_empty:
            current = MultiPolygon()
            break
        current = _to_multi(inner)

    infill = current if not current.is_empty else None
    return LayerRegions(wall_centerlines=centerlines, infill_region=_to_multi(infill) if infill else None)


def _to_multi(geom) -> MultiPolygon:
    if geom is None or geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    # GeometryCollection 等情况：提取所有 Polygon
    polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return MultiPolygon(polys) if polys else MultiPolygon()
