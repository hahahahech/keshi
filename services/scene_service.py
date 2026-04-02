"""
Application service for scene state and rendering orchestration.
"""

from __future__ import annotations

from core.project import GeologicalProject
from core.scene import MeshSceneObject
from rendering.render_manager import RenderManager


class SceneService:
    def __init__(self, project: GeologicalProject | None = None, render_manager: RenderManager | None = None):
        self.project = project or GeologicalProject()
        self.render_manager = render_manager or RenderManager()

    def set_plotter(self, plotter):
        self.render_manager.set_plotter(plotter)

    def add_object(self, scene_object, render: bool = True):
        scene_object.attach_render_manager(self.render_manager)
        self.project.scene.add_object(scene_object)
        if render and self.render_manager.plotter is not None:
            self.render_manager.render_object(scene_object)
        return scene_object

    def create_model_object(self, dataset):
        return MeshSceneObject(dataset=dataset, object_type="model")

    def add_dataset_as_model(self, dataset, render: bool = True):
        return self.add_object(self.create_model_object(dataset), render=render)

    def get_object(self, object_id: str):
        return self.project.scene.get_object(object_id)

    def get_objects_by_type(self, object_type: str):
        return self.project.scene.get_objects_by_type(object_type)

    def all_objects(self):
        return self.project.scene.all_objects()

    def remove_object(self, object_id: str, cleanup: bool = True):
        scene_object = self.project.scene.remove_object(object_id)
        if scene_object is not None and cleanup:
            scene_object.cleanup(plotter=self.render_manager.plotter)
        return scene_object
