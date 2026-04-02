"""
VTK / PyVista rendering coordination layer.
"""

from __future__ import annotations

import pyvista as pv


class RenderManager:
    def __init__(self, plotter=None):
        self.plotter = plotter

    def set_plotter(self, plotter):
        self.plotter = plotter

    def render_object(self, scene_object):
        plotter = self.plotter
        if plotter is None:
            raise ValueError("A plotter must be attached before rendering scene objects.")

        self.remove_object(scene_object, plotter=plotter)

        mesh = getattr(scene_object, "mesh", None)
        if mesh is None:
            raise ValueError(f"Scene object {scene_object.object_id} has no renderable mesh.")

        actor = plotter.add_mesh(
            mesh,
            name=scene_object.object_id,
            color=scene_object.color,
            opacity=scene_object.opacity,
            smooth_shading=isinstance(mesh, pv.PolyData),
            show_edges=False,
        )

        scene_object.actor = actor
        scene_object.actors = [actor]
        scene_object.attach_render_manager(self)
        self.apply_style(scene_object)
        return actor

    def apply_style(self, scene_object, highlight: bool = False):
        for actor in list(getattr(scene_object, "actors", [])):
            if actor is None:
                continue

            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(scene_object.visible)

            if not hasattr(actor, "GetProperty"):
                continue

            prop = actor.GetProperty()
            prop.SetOpacity(scene_object.opacity)
            prop.SetColor(*scene_object.color)

            mode = scene_object.render_mode
            if mode == "wireframe":
                prop.SetRepresentationToWireframe()
                prop.SetLineWidth(2.0 if highlight else 1.0)
                prop.SetPointSize(1.0)
            elif mode == "points":
                prop.SetRepresentationToPoints()
                prop.SetPointSize(8.0 if highlight else 5.0)
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
            for actor in list(getattr(scene_object, "actors", [])):
                if actor is None:
                    continue
                try:
                    active_plotter.remove_actor(actor)
                except Exception:
                    pass

        scene_object.actor = None
        scene_object.actors = []
