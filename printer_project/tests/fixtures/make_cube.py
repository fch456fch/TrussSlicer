"""生成测试用长方体 STL。"""
import trimesh
from pathlib import Path

box = trimesh.creation.box(extents=[50, 28, 50])
out = Path(__file__).parent / "box_50x28x50.stl"
box.export(str(out))
print(f"已写入 {out}")
