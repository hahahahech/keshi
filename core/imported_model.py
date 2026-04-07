"""
导入网格模型的兼容适配器。
"""

from __future__ import annotations

from core.datasets import MeshDataset
from core.scene import MeshSceneObject


class ImportedModel(MeshSceneObject):
    """基于新场景架构的旧版模型兼容对象。"""

    def __init__(self, mesh, file_path: str, name: str | None = None):
        dataset = MeshDataset(data=mesh, source_path=file_path, name=name)
        super().__init__(dataset=dataset, name=dataset.name, object_type="dataset")
