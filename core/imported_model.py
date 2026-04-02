"""
通用导入模型对象
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv


class ImportedModel:
    """用于场景树和三维视图的轻量模型对象。"""

    _counter = 0

    def __init__(self, mesh, file_path: str, name: str | None = None):
        self.mesh = self._normalize_mesh(mesh)
        self.file_path = file_path
        self.name = name or Path(file_path).stem
        self.visible = True
        self.opacity = 1.0
        self.color = (0.75, 0.78, 0.85)
        self.actor = None
        self.actors = []
        self.render_mode = "surface"

        ImportedModel._counter += 1
        self.model_id = f"model_{ImportedModel._counter}"
        self.id = self.model_id

    @staticmethod
    def _normalize_mesh(mesh):
        if isinstance(mesh, pv.MultiBlock):
            if mesh.n_blocks == 0:
                raise ValueError("文件中没有可显示的几何数据")

            try:
                combined = mesh.combine(merge_points=False)
            except Exception as exc:
                raise ValueError(f"无法合并多块数据: {exc}") from exc

            if combined is None or combined.n_points == 0:
                raise ValueError("文件中没有可显示的几何数据")
            return combined

        if mesh is None or mesh.n_points == 0:
            raise ValueError("文件中没有可显示的几何数据")
        return mesh

    def create_actor(self, plotter):
        if self.actor is not None:
            try:
                plotter.remove_actor(self.actor)
            except Exception:
                pass

        self.actor = plotter.add_mesh(
            self.mesh,
            name=self.model_id,
            color=self.color,
            opacity=self.opacity,
            smooth_shading=isinstance(self.mesh, pv.PolyData),
            show_edges=False,
        )
        if hasattr(self.actor, "SetVisibility"):
            self.actor.SetVisibility(self.visible)
        self.actors = [self.actor]
        return self.actor

    def set_render_mode(self, mode: str, highlight: bool = False):
        self.render_mode = mode
        if self.actor is None or not hasattr(self.actor, "GetProperty"):
            return

        prop = self.actor.GetProperty()
        if mode == "wireframe":
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(2.0 if highlight else 1.0)
        else:
            prop.SetRepresentationToSurface()
            prop.SetColor(*self.color)
            prop.SetLineWidth(1.0)

    def cleanup(self, plotter=None):
        if plotter is not None and self.actor is not None:
            try:
                plotter.remove_actor(self.actor)
            except Exception:
                pass
        self.actor = None
        self.actors = []
