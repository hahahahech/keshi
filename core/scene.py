"""
场景图基础对象定义。
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
        object_type: str = "dataset",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_object_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.dataset = dataset
        self.object_type = object_type
        self.object_id = object_id or _next_object_id(object_type)
        self.id = self.object_id
        self.style = (style or RenderStyle()).normalized()
        self.metadata = metadata or {}
        self.source_object_id = source_object_id
        self.parameters = parameters or {}
        self.actor = None
        self.actors: list[Any] = []
        self._render_manager = None
        if self.style.scalar_name is None and self.dataset is not None:
            self.style.scalar_name = self.dataset.active_scalar
        if self.style.clim is None and self.dataset is not None:
            self.style.clim = self.dataset.get_scalar_range(self.style.scalar_name)

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
    def color(self) -> tuple[float, float, float] | None:
        return self.style.color

    @color.setter
    def color(self, value: tuple[float, float, float] | None):
        self.style.color = value

    @property
    def render_mode(self) -> str:
        return self.style.render_mode

    @render_mode.setter
    def render_mode(self, value: str):
        self.style.render_mode = value

    @property
    def active_scalar(self) -> str | None:
        return self.style.scalar_name or (self.dataset.active_scalar if self.dataset else None)

    @property
    def data(self):
        return self.dataset.data if self.dataset is not None else None

    @property
    def mesh(self):
        return self.data

    @property
    def bounds(self):
        return self.dataset.bounds if self.dataset is not None else None

    def set_active_scalar(self, scalar_name: str | None):
        self.style.scalar_name = scalar_name
        if self.dataset is not None:
            self.dataset.set_active_scalar(scalar_name)
            if self.style.clim is None:
                self.style.clim = self.dataset.get_scalar_range(scalar_name)

    def attach_render_manager(self, render_manager):
        self._render_manager = render_manager

    def create_actor(self, plotter):
        from rendering.render_manager import RenderManager

        manager = self._render_manager
        if manager is None or manager.plotter is not plotter:
            manager = RenderManager(plotter)
            self.attach_render_manager(manager)
        return manager.render_object(self)

    def rerender(self):
        if self._render_manager is not None:
            self._render_manager.render_object(self)

    def set_render_mode(self, mode: str, highlight: bool = False):
        self.render_mode = mode
        if self._render_manager is not None:
            self._render_manager.apply_style(self, highlight=highlight)

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

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "object_id": self.object_id,
            "name": self.name,
            "object_type": self.object_type,
            "style": self.style.to_dict(),
            "metadata": self.metadata,
            "source_object_id": self.source_object_id,
            "parameters": self.parameters,
        }
        if self.dataset is not None:
            payload["dataset"] = self.dataset.to_dict()
        return payload


class DatasetSceneObject(SceneObject):
    def __init__(
        self,
        dataset: BaseDataset,
        name: str | None = None,
        object_type: str = "dataset",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_object_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name or dataset.name,
            dataset=dataset,
            object_type=object_type,
            style=style,
            object_id=object_id,
            metadata=metadata,
            source_object_id=source_object_id,
            parameters=parameters,
        )

        if object_type == "dataset":
            self.dataset_id = self.object_id

    @property
    def file_path(self) -> str:
        return self.dataset.source_path


class MeshSceneObject(DatasetSceneObject):
    """兼容旧版以网格为中心的场景对象别名。"""

    def __init__(
        self,
        dataset: MeshDataset,
        name: str | None = None,
        object_type: str = "dataset",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_object_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        super().__init__(
            dataset=dataset,
            name=name,
            object_type=object_type,
            style=style,
            object_id=object_id,
            metadata=metadata,
            source_object_id=source_object_id,
            parameters=parameters,
        )


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

    def clear(self) -> list[SceneObject]:
        objects = self.all_objects()
        self._objects.clear()
        self._order.clear()
        return objects

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
