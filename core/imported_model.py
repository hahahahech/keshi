"""
Compatibility adapter for imported mesh models.
"""

from __future__ import annotations

from core.datasets import MeshDataset
from core.scene import MeshSceneObject


class ImportedModel(MeshSceneObject):
    """Backward-compatible model object backed by the new scene architecture."""

    def __init__(self, mesh, file_path: str, name: str | None = None):
        dataset = MeshDataset(mesh=mesh, source_path=file_path, name=name)
        super().__init__(dataset=dataset, name=dataset.name, object_type="model")
