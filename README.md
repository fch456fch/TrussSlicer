# truss-slicer

桁架结构 3D 打印路径规划切片器。从 STL/OBJ 模型生成包含 3D 桁架填充的打印路径预览。

## 安装

```bash
pip install -e .
```

## 使用

```bash
truss-slicer model.stl --layer-h 0.2 --truss-h 5 --preview out.html
```

打开 `out.html` 在浏览器中查看 3D 路径。

## 模块

- `mesh_loader` —— STL/OBJ 加载
- `slicer` —— 平面切片，模型 → 每层 2D 多边形
- `layer_classifier` —— 层归类（顶/底/锚定/桁架）
- `region_builder` —— 外壳 offset，分离 wall / infill 区域
- `truss_grid` —— 桁架网格生成（节点高低 Z 属性）
- `snake_planner` —— 蛇形路径（DFS 深度预判 + 最短弧连接）
- `preview` —— plotly 3D 可视化
- `cli` —— 命令行入口
