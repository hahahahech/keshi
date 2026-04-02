"""
Dataset abstractions for geological visualization data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyvista as pv


@dataclass
class DatasetMetadata:
    source_path: str = ""
    source_name: str = ""
    dataset_type: str = "mesh"
    attributes: dict[str, Any] = field(default_factory=dict)


class BaseDataset:
    def __init__(self, name: str, metadata: DatasetMetadata | None = None):
        self.name = name
        self.metadata = metadata or DatasetMetadata(source_name=name)


class MeshDataset(BaseDataset):
    def __init__(
        self,
        mesh: pv.DataSet,
        source_path: str = "",
        name: str | None = None,
        metadata: DatasetMetadata | None = None,
    ):
        dataset_name = name or (Path(source_path).stem if source_path else "mesh_dataset")
        metadata = metadata or DatasetMetadata(
            source_path=source_path,
            source_name=dataset_name,
            dataset_type="mesh",
        )
        super().__init__(name=dataset_name, metadata=metadata)
        self.mesh = self._normalize_mesh(mesh)

    @staticmethod
    def _normalize_mesh(mesh: pv.DataSet) -> pv.DataSet:
        if isinstance(mesh, pv.MultiBlock):
            if mesh.n_blocks == 0:
                raise ValueError("The file does not contain displayable geometry.")

            try:
                combined = mesh.combine(merge_points=False)
            except Exception as exc:
                raise ValueError(f"Unable to combine multiblock dataset: {exc}") from exc

            if combined is None or combined.n_points == 0:
                raise ValueError("The file does not contain displayable geometry.")
            return combined

        if mesh is None or mesh.n_points == 0:
            raise ValueError("The file does not contain displayable geometry.")
        return mesh

    @property
    def source_path(self) -> str:
        return self.metadata.source_path

    @property
    def bounds(self):
        return self.mesh.bounds
