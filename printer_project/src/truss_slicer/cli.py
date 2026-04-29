"""命令行入口。"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from shapely.geometry import LinearRing

from . import mesh_loader, preview, slicer
from .layer_classifier import classify_layers
from .region_builder import build_layer_regions
from .snake_planner import generate_grid_lines, generate_parallel_lines, plan_snake_paths, GridLine
from .truss_grid import generate_truss_grid


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--layer-h", "layer_h", default=0.2, show_default=True, type=float, help="层高 (mm)")
@click.option("--truss-h", "truss_h", default=5.0, show_default=True, type=float, help="桁架单元高度 (mm)")
@click.option("--wall-count", default=3, show_default=True, type=int, help="外壳圈数")
@click.option("--line-width", default=0.4, show_default=True, type=float, help="挤出线宽 (mm)")
@click.option("--infill-spacing", default=5.0, show_default=True, type=float, help="桁架/填充网格间距 (mm)")
@click.option("--infill-angle", default=45.0, show_default=True, type=float, help="填充角度 (度)")
@click.option("--top-layers", default=3, show_default=True, type=int)
@click.option("--bottom-layers", default=3, show_default=True, type=int)
@click.option("--no-walls", is_flag=True, default=False, help="不生成外壳（仅填充）")
@click.option(
    "--preview",
    "preview_path",
    default="preview.html",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="输出 HTML 预览路径",
)
def main(
    input_path: Path,
    layer_h: float,
    truss_h: float,
    wall_count: int,
    line_width: float,
    infill_spacing: float,
    infill_angle: float,
    top_layers: int,
    bottom_layers: int,
    no_walls: bool,
    preview_path: Path,
) -> None:
    """加载模型，切片，生成桁架填充路径，输出 3D 预览。"""
    click.echo(f"[1/5] 加载 {input_path}")
    mesh = mesh_loader.load_mesh(input_path)
    info = mesh_loader.mesh_info(mesh)
    click.echo(f"      尺寸 {info['size']} mm, 面数 {info['n_faces']}, 水密 {info['is_watertight']}")

    click.echo(f"[2/5] 平面切片 (层高 {layer_h} mm)")
    layers = slicer.slice_mesh(mesh, layer_h)
    click.echo(f"      {len(layers)} 层")

    truss_cell_layers = max(2, round(truss_h / layer_h))
    click.echo(f"[3/5] 层归类 (桁架单元 {truss_cell_layers} 层)")
    labels = classify_layers(len(layers), top_layers, bottom_layers, truss_cell_layers)
    label_map = {lb.index: lb for lb in labels}

    counts = {}
    for lb in labels:
        counts[lb.kind] = counts.get(lb.kind, 0) + 1
    click.echo(f"      {counts}")

    click.echo(f"[4/5] 生成路径")
    all_layer_paths: list[preview.LayerPaths] = []
    # truss_body_high 需要延迟到目标层侧墙之后插入
    deferred_high: dict[int, list[preview.LayerPaths]] = {}

    for layer in layers:
        lb = label_map.get(layer.index)
        if lb is None:
            continue

        # 外壳 + 填充区域
        effective_walls = 0 if no_walls else wall_count
        regions = build_layer_regions(layer.polygons, effective_walls, line_width)

        # 外壳路径（中心线）
        if not no_walls:
            for wall_mp in regions.wall_centerlines:
                wp = preview.multipolygon_to_layer_paths(wall_mp, layer.z, layer.index, kind="wall")
                all_layer_paths.append(wp)

        # 插入延迟的 truss_body_high（需要先有侧墙才能挂）
        if layer.index in deferred_high:
            all_layer_paths.extend(deferred_high.pop(layer.index))

        if regions.infill_region is None or regions.infill_region.is_empty:
            continue

        # 收集所有边界环（外环 + 内环）
        boundaries = []
        for poly in regions.infill_region.geoms:
            boundaries.append(LinearRing(poly.exterior.coords))
            for interior in poly.interiors:
                boundaries.append(LinearRing(interior.coords))

        if not boundaries:
            continue

        if lb.kind in ("bottom_solid", "top_solid"):
            solid_angle = 45.0 if (layer.index % 2 == 0) else 135.0
            grid = generate_parallel_lines(regions.infill_region, spacing=line_width, angle=solid_angle)
            paths = plan_snake_paths(grid, boundaries)
            kind = lb.kind

        elif lb.kind == "truss_anchor":
            grid = generate_grid_lines(regions.infill_region, spacing=infill_spacing, angle=infill_angle)
            paths = plan_snake_paths(grid, boundaries)
            kind = "truss_anchor"

        elif lb.kind == "sparse_infill":
            # 余数层：20% 密度传统网格（间距 = line_width / 0.2 = 5 倍线宽）
            sparse_spacing = line_width / 0.2
            sparse_angle = 45.0 if (layer.index % 2 == 0) else 135.0
            grid = generate_parallel_lines(regions.infill_region, spacing=sparse_spacing, angle=sparse_angle)
            paths = plan_snake_paths(grid, boundaries)
            kind = "sparse_infill"

        elif lb.kind == "truss_body_start":
            z_low = layer.z
            z_high = z_low + truss_h - 2 * layer_h
            truss_result = generate_truss_grid(
                regions.infill_region, spacing=infill_spacing,
                z_low=z_low, z_high=z_high,
                cell_index=lb.cell_index,
                angle=infill_angle,
            )
            # 底部蛇形
            if truss_result.bottom_lines:
                bottom_paths = plan_snake_paths(truss_result.bottom_lines, boundaries, arc_z=z_low)
                lp_low = preview.LayerPaths(index=layer.index, z=layer.z, kind="truss_body_low")
                for p in bottom_paths:
                    if p.shape[1] == 2:
                        zs = np.full((len(p), 1), z_low)
                        p = np.hstack([p, zs])
                    lp_low.paths.append(p)
                all_layer_paths.append(lp_low)
            # 顶部蛇形 — 延迟到锚固层前一层的侧墙之后
            if truss_result.top_lines:
                top_paths = plan_snake_paths(truss_result.top_lines, boundaries, arc_z=z_high)
                anchor_idx = layer.index + truss_cell_layers - 2
                anchor_layer = layers[anchor_idx] if anchor_idx < len(layers) else layer
                lp_high = preview.LayerPaths(index=anchor_idx, z=anchor_layer.z, kind="truss_body_high")
                for p in top_paths:
                    if p.shape[1] == 2:
                        zs = np.full((len(p), 1), z_high)
                        p = np.hstack([p, zs])
                    lp_high.paths.append(p)
                deferred_high.setdefault(anchor_idx, []).append(lp_high)
            continue
        else:
            continue

        lp = preview.LayerPaths(index=layer.index, z=layer.z, kind=kind)
        for p in paths:
            if p.shape[1] == 2:
                zs = np.full((len(p), 1), layer.z)
                p = np.hstack([p, zs])
            lp.paths.append(p)
        all_layer_paths.append(lp)

    click.echo(f"[5/5] 渲染预览 → {preview_path}")
    preview.render_html(all_layer_paths, str(preview_path), title=input_path.name)
    click.echo("完成。")


if __name__ == "__main__":
    main()
