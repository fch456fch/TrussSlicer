"""层归类：顶 / 底 / 锚定 / 桁架。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerLabel:
    index: int
    kind: str  # 'bottom_solid' | 'top_solid' | 'sparse_infill' | 'truss_anchor' | 'truss_body_start' | 'truss_body_skip'
    cell_index: int = 0  # 桁架单元编号（仅 truss_anchor / truss_body 有意义）


def classify_layers(
    n_layers: int,
    top_layers: int = 4,
    bottom_layers: int = 4,
    truss_cell_layers: int = 25,
) -> list[LayerLabel]:
    """
    对 n_layers 层进行归类。

    桁架单元 = (truss_cell_layers - 1) 层桁架本体 + 1 层锚定网格。
    中间区域不能被整除时，余数对半分配：一半放入底层实心区与桁架之间，
    另一半放入桁架与顶层实心区之间。
    """
    if n_layers <= 0:
        return []

    labels: list[LayerLabel] = []

    # 底层实心
    actual_bottom = min(bottom_layers, n_layers)
    for i in range(actual_bottom):
        labels.append(LayerLabel(index=i, kind="bottom_solid"))

    # 计算中间有多少层可用于桁架
    remaining = n_layers - actual_bottom - top_layers
    if remaining <= 0:
        for i in range(actual_bottom, n_layers):
            labels.append(LayerLabel(index=i, kind="top_solid"))
        return labels

    # 桁架区域
    n_cells = remaining // truss_cell_layers
    truss_used = n_cells * truss_cell_layers
    extra = remaining - truss_used

    # 余数对半分配
    extra_bottom = extra // 2
    extra_top = extra - extra_bottom

    # 底层余数：紧接在底层实心之后，桁架之前
    cell_start = actual_bottom + extra_bottom

    for c in range(n_cells):
        body_count = truss_cell_layers - 1
        idx = cell_start + c * truss_cell_layers
        labels.append(LayerLabel(index=idx, kind="truss_body_start", cell_index=c))
        for j in range(1, body_count):
            idx = cell_start + c * truss_cell_layers + j
            labels.append(LayerLabel(index=idx, kind="truss_body_skip", cell_index=c))
        anchor_idx = cell_start + c * truss_cell_layers + body_count
        labels.append(LayerLabel(index=anchor_idx, kind="truss_anchor", cell_index=c))

    # 底层余数层：底层实心区与桁架之间
    for i in range(actual_bottom, actual_bottom + extra_bottom):
        labels.append(LayerLabel(index=i, kind="sparse_infill"))

    # 顶层余数层：桁架与顶层实心区之间
    top_start = cell_start + truss_used
    for i in range(top_start, top_start + extra_top):
        labels.append(LayerLabel(index=i, kind="sparse_infill"))

    # 顶层实心
    for i in range(top_start + extra_top, n_layers):
        labels.append(LayerLabel(index=i, kind="top_solid"))

    return labels
