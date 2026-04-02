"""
Scene graph primitives.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import count
from typing import Any

from core.datasets import BaseDataset, MeshDataset
from core.styles import RenderStyle


_OBJECT_COUNTERS: defaultdict[str, count] = defaultdict(lambda: count(1))


def _next_object_id(object_type: str) -> str:
    prefix = object_type or "object"
    return f"{prefix}_{next(_OBJECT_COUNTERS[prefix])}"


class SceneObject:
    def __init__(
        self,
        name: str,
        dataset: BaseDataset | None = None,
        object_type: str = "object",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.dataset = dataset
        self.object_type = object_type
        self.object_id = object_id or _next_object_id(object_type)
        self.id = self.object_id
        self.style = (style or RenderStyle()).normalized()
        self.metadata = metadata or {}
        self.actor = None
        self.actors: list[Any] = []
        self._render_manager = None

    @property
    def visible(self) -> bool:
        return self.style.visible

    @visible.setter
    def visible(self, value: bool):
        self.style.visible = bool(value)

    @property
    def opacity(self) -> float:
        return self.style.opacity

    @opacity.setter
    def opacity(self, value: float):
        self.style.opacity = max(0.0, min(float(value), 1.0))

    @property
    def color(self) -> tuple[float, float, float]:
        return self.style.color

    @color.setter
    def color(self, value: tuple[float, float, float]):
        self.style.color = value

    @property
    def render_mode(self) -> str:
        return self.style.render_mode

    @render_mode.setter
    def render_mode(self, value: str):
        self.style.render_mode = value

    def attach_render_manager(self, render_manager):
        self._render_manager = render_manager

    def create_actor(self, plotter):
        from rendering.render_manager import RenderManager

        manager = self._render_manager
        if manager is None or manager.plotter is not plotter:
            manager = RenderManager(plotter)
            self.attach_render_manager(manager)
        return manager.render_object(self)

    def set_render_mode(self, mode: str, highlight: bool = False):
        self.render_mode = mode
        if self._render_manager is not None:
            self._render_manager.apply_style(self, highlight=highlight)
            return

        for actor in self.actors:
            if not hasattr(actor, "GetProperty"):
                continue
            prop = actor.GetProperty()
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
                prop.SetColor(*self.color)
                prop.SetPointSize(1.0)
                prop.SetLineWidth(1.0)

    def cleanup(self, plotter=None):
        if self._render_manager is not None:
            self._render_manager.remove_object(self, plotter=plotter)
            return

        if plotter is not None:
            for actor in list(self.actors):
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
        self.actor = None
        self.actors = []


class MeshSceneObject(SceneObject):
    def __init__(
        self,
        dataset: MeshDataset,
        name: str | None = None,
        object_type: str = "model",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name or dataset.name,
            dataset=dataset,
            object_type=object_type,
            style=style,
            object_id=object_id,
            metadata=metadata,
        )

        if object_type == "model":
            self.model_id = self.object_id

    @property
    def mesh(self):
        return self.dataset.mesh

    @property
    def file_path(self) -> str:
        return self.dataset.source_path


class SceneGraph:
    def __init__(self):
        self._objects: dict[str, SceneObject] = {}
        self._order: list[str] = []

    def add_object(self, scene_object: SceneObject) -> SceneObject:
        self._objects[scene_object.object_id] = scene_object
        if scene_object.object_id not in self._order:
            self._order.append(scene_object.object_id)
        return scene_object

    def remove_object(self, object_id: str) -> SceneObject | None:
        removed = self._objects.pop(object_id, None)
        if object_id in self._order:
            self._order.remove(object_id)
        return removed

    def get_object(self, object_id: str) -> SceneObject | None:
        return self._objects.get(object_id)

    def get_objects_by_type(self, object_type: str) -> list[SceneObject]:
        return [
            self._objects[object_id]
            for object_id in self._order
            if self._objects[object_id].object_type == object_type
        ]

    def all_objects(self) -> list[SceneObject]:
        return [self._objects[object_id] for object_id in self._order]
