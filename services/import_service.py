"""
Application service for loading external data into datasets / scene objects.
"""

from __future__ import annotations

import pyvista as pv

from core.datasets import MeshDataset


class ImportService:
    def load_mesh_dataset(self, file_path: str) -> MeshDataset:
        mesh = pv.read(file_path)
        return MeshDataset(mesh=mesh, source_path=file_path)

    def import_models(self, file_paths, scene_service=None):
        imported = []
        failures = []

        for file_path in file_paths:
            try:
                dataset = self.load_mesh_dataset(file_path)
                if scene_service is None:
                    imported.append(dataset)
                else:
                    imported.append(scene_service.add_dataset_as_model(dataset, render=True))
            except Exception as exc:
                failures.append((file_path, exc))

        return imported, failures
