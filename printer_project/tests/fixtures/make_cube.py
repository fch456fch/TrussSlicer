"""生成测试用立方体 STL。"""
import trimesh
from pathlib import Path

cube = trimesh.creation.box(extents=[20, 20, 20])
out = Path(__file__).parent / "cube_20mm.stl"
cube.export(str(out))
print(f"已写入 {out}")
