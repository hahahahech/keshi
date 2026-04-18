"""
VTK / PyVista 渲染协调层。
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


class RenderManager:
    def __init__(self, plotter=None):
        self.plotter = plotter
        self._scalar_bar_owner_id: str | None = None

    def set_plotter(self, plotter):
        self.plotter = plotter

    def render_object(self, scene_object):
        plotter = self.plotter
        if plotter is None:
            raise ValueError("渲染场景对象前必须先绑定绘图器。")

        self.remove_object(scene_object, plotter=plotter)

        dataset = getattr(scene_object, "dataset", None)
        if dataset is None:
            raise ValueError(f"场景对象 {scene_object.object_id} 没有可渲染的数据集。")

        scalar_name = scene_object.active_scalar
        render_mode = scene_object.render_mode
        show_scalar_bar = bool(scene_object.style.show_scalar_bar and scalar_name)
        self._prepare_scalar_bar(scene_object.object_id, show_scalar_bar)
        prepared = self._prepare_display_data(scene_object)
        actors = []

        if render_mode == "volume" and dataset.is_regular_grid and scene_object.style.threshold_range is None:
            volume_data, scalar_name = dataset.get_volume_render_data(scalar_name)
            if dataset.metadata.preview_mode:
                prepared = self._build_preview_data(volume_data)
            else:
                actors.append(
                    plotter.add_volume(
                        volume_data,
                        scalars=scalar_name,
                        cmap=scene_object.style.colormap,
                        clim=scene_object.style.clim,
                        opacity=[value * scene_object.opacity for value in scene_object.style.opacity_curve],
                        show_scalar_bar=show_scalar_bar,
                        name=scene_object.object_id,
                    )
                )

        if not actors:
            actors.extend(
                self._render_as_mesh(
                    scene_object=scene_object,
                    data=prepared,
                    scalar_name=scalar_name,
                    render_mode="surface" if render_mode == "volume" else render_mode,
                )
            )

        scene_object.actor = actors[0] if actors else None
        scene_object.actors = actors
        scene_object.attach_render_manager(self)
        self.apply_style(scene_object)
        return scene_object.actor

    def apply_style(self, scene_object, highlight: bool = False):
        for actor in list(getattr(scene_object, "actors", [])):
            if actor is None:
                continue

            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(scene_object.visible)

            if not hasattr(actor, "GetProperty"):
                continue

            prop = actor.GetProperty()

            if actor.__class__.__name__.lower().endswith("volume"):
                continue

            prop.SetOpacity(scene_object.opacity)
            if scene_object.color is not None and scene_object.active_scalar is None:
                prop.SetColor(*scene_object.color)

            mode = scene_object.render_mode
            if mode == "wireframe":
                prop.SetRepresentationToWireframe()
                prop.SetLineWidth(2.0 if highlight else 1.0)
                prop.SetPointSize(1.0)
            elif mode == "points":
                prop.SetRepresentationToPoints()
                prop.SetPointSize(8.0 if highlight else 6.0)
                prop.SetLineWidth(1.0)
            else:
                prop.SetRepresentationToSurface()
                prop.SetLineWidth(1.0)
                prop.SetPointSize(1.0)

        if self.plotter is not None:
            self.plotter.render()

    def remove_object(self, scene_object, plotter=None):
        active_plotter = plotter or self.plotter
        if active_plotter is not None:
            if self._scalar_bar_owner_id == scene_object.object_id:
                self._clear_scalar_bars(active_plotter)
                self._scalar_bar_owner_id = None
            for actor in list(getattr(scene_object, "actors", [])):
                if actor is None:
                    continue
                try:
                    active_plotter.remove_actor(actor)
                except Exception:
                    pass

        scene_object.actor = None
        scene_object.actors = []

    def _prepare_display_data(self, scene_object):
        dataset = scene_object.dataset
        data, scalar_name = dataset.get_render_data(scene_object.active_scalar)
        if dataset.metadata.preview_mode and scene_object.render_mode != "points":
            return self._build_preview_data(data)

        if scene_object.style.threshold_range is not None and scalar_name is not None:
            try:
                data = data.threshold(scene_object.style.threshold_range, scalars=scalar_name)
            except Exception:
                pass

        if isinstance(data, pv.PolyData):
            return data

        if scene_object.object_type == "dataset" and scene_object.render_mode == "surface":
            try:
                return data.extract_surface()
            except Exception:
                return data

        return data

    def _build_preview_data(self, data: pv.DataSet):
        if isinstance(data, pv.ImageData):
            bounds = np.array(data.bounds, dtype=float)
            center = (
                float((bounds[0] + bounds[1]) / 2.0),
                float((bounds[2] + bounds[3]) / 2.0),
                float((bounds[4] + bounds[5]) / 2.0),
            )
            return data.slice_orthogonal(x=center[0], y=center[1], z=center[2])
        if hasattr(data, "extract_surface"):
            try:
                return data.extract_surface()
            except Exception:
                return data
        return data

    def _render_as_mesh(self, scene_object, data, scalar_name: str | None, render_mode: str) -> list:
        blocks = list(self._iter_blocks(data))
        actors = []
        preference = scene_object.dataset.get_scalar_association(scalar_name)
        show_scalar_bar = bool(scene_object.style.show_scalar_bar and scalar_name)

        for index, block in enumerate(blocks):
            if getattr(block, "n_points", 0) == 0:
                continue
            kwargs = {
                "name": f"{scene_object.object_id}_{index}",
                "opacity": scene_object.opacity,
                "cmap": scene_object.style.colormap,
                "clim": scene_object.style.clim,
                "show_scalar_bar": bool(show_scalar_bar and index == 0),
                "smooth_shading": isinstance(block, pv.PolyData),
                "show_edges": False,
                "render_points_as_spheres": render_mode == "points",
                "point_size": 6,
            }
            if scalar_name is not None:
                kwargs["scalars"] = scalar_name
                kwargs["preference"] = preference
            elif scene_object.color is not None:
                kwargs["color"] = scene_object.color
            actor = self.plotter.add_mesh(block, **kwargs)
            actors.append(actor)
        return actors

    def _iter_blocks(self, data):
        if isinstance(data, pv.MultiBlock):
            for block in data:
                if block is not None:
                    yield block
            return
        yield data

    def _prepare_scalar_bar(self, object_id: str, show_scalar_bar: bool):
        if self.plotter is None:
            return
        if show_scalar_bar:
            self._clear_scalar_bars(self.plotter)
            self._scalar_bar_owner_id = object_id
            return
        if self._scalar_bar_owner_id == object_id:
            self._clear_scalar_bars(self.plotter)
            self._scalar_bar_owner_id = None

    def _clear_scalar_bars(self, plotter):
        for title in list(getattr(plotter, "scalar_bars", {}).keys()):
            try:
                plotter.remove_scalar_bar(title=title, render=False)
            except Exception:
                pass
