"""
外部数据导入为数据集与场景对象的应用服务。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from core.datasets import (
    BaseDataset,
    DatasetMetadata,
    ImportSpec,
    PointSetDataset,
    RegularGridDataset,
    SurfaceDataset,
    UnstructuredGridDataset,
    create_dataset_from_pyvista,
)


TEXT_EXTENSIONS = {".csv", ".xyz", ".dat", ".txt"}


class ImportService:
    def load_dataset(self, file_path: str, import_spec: ImportSpec | None = None) -> BaseDataset:
        extension = Path(file_path).suffix.lower()
        if extension in TEXT_EXTENSIONS:
            return self.load_text_dataset(file_path, import_spec=import_spec)
        mesh = pv.read(file_path)
        return self._wrap_loaded_dataset(mesh, file_path=file_path, import_spec=import_spec)

    def import_files(
        self,
        file_paths: list[str],
    ) -> tuple[list[BaseDataset], list[tuple[str, Exception]]]:
        imported: list[BaseDataset] = []
        failures: list[tuple[str, Exception]] = []
        for file_path in file_paths:
            try:
                imported.append(self.load_dataset(file_path))
            except Exception as exc:
                failures.append((file_path, exc))
        return imported, failures

    def import_models(self, file_paths, scene_service=None):
        imported = []
        failures = []

        for file_path in file_paths:
            try:
                dataset = self.load_dataset(file_path)
                if scene_service is None:
                    imported.append(dataset)
                else:
                    imported.append(scene_service.add_dataset(dataset, render=True))
            except Exception as exc:
                failures.append((file_path, exc))

        return imported, failures

    def load_text_dataset(
        self,
        file_path: str,
        import_spec: ImportSpec | None = None,
    ) -> BaseDataset:
        spec = import_spec or self._infer_import_spec(file_path)
        numeric_rows, headers = self._read_numeric_rows(file_path, spec)
        if not numeric_rows:
            raise ValueError("文件中没有找到有效的数值数据行。")

        column_arrays = {
            header: np.array([row[index] for row in numeric_rows], dtype=float)
            for index, header in enumerate(headers)
        }
        x = column_arrays[spec.x_column]
        y = column_arrays[spec.y_column]
        z = column_arrays[spec.z_column]
        scalar_columns = spec.scalar_columns or [
            header for header in headers if header not in {spec.x_column, spec.y_column, spec.z_column}
        ]

        if spec.nodata is not None:
            for column in scalar_columns:
                values = column_arrays[column]
                values[np.isclose(values, spec.nodata)] = np.nan

        if spec.is_regular_grid and self._can_make_image_data(x, y, z, spec):
            image = self._build_image_data(x, y, z, scalar_columns, column_arrays, spec)
            metadata = DatasetMetadata(
                source_path=file_path,
                source_name=Path(file_path).stem,
                dataset_type="regular_grid",
                source_schema={
                    "headers": headers,
                    "row_count": len(numeric_rows),
                    "loader": "text",
                },
                units=spec.units,
                nodata=spec.nodata,
            )
            return RegularGridDataset(
                data=image,
                source_path=file_path,
                name=Path(file_path).stem,
                metadata=metadata,
                import_spec=spec,
            )

        points = np.column_stack([x, y, z])
        poly = pv.PolyData(points)
        for column in scalar_columns:
            poly.point_data[column] = column_arrays[column]
        metadata = DatasetMetadata(
            source_path=file_path,
            source_name=Path(file_path).stem,
            dataset_type="point_set",
            source_schema={
                "headers": headers,
                "row_count": len(numeric_rows),
                "loader": "text",
            },
            units=spec.units,
            nodata=spec.nodata,
        )
        return PointSetDataset(
            data=poly,
            source_path=file_path,
            name=Path(file_path).stem,
            metadata=metadata,
            import_spec=spec,
        )

    def _wrap_loaded_dataset(
        self,
        mesh: pv.DataSet,
        file_path: str,
        import_spec: ImportSpec | None = None,
    ) -> BaseDataset:
        source_name = Path(file_path).stem
        metadata = DatasetMetadata(
            source_path=file_path,
            source_name=source_name,
            dataset_type="dataset",
            source_schema={"loader": "pyvista"},
        )

        if isinstance(mesh, pv.RectilinearGrid):
            converted = self._rectilinear_to_image_if_uniform(mesh)
            if converted is not None:
                return RegularGridDataset(
                    data=converted,
                    source_path=file_path,
                    name=source_name,
                    metadata=metadata,
                    import_spec=import_spec,
                )

        if isinstance(mesh, pv.StructuredGrid):
            converted = self._structured_to_image_if_uniform(mesh)
            if converted is not None:
                return RegularGridDataset(
                    data=converted,
                    source_path=file_path,
                    name=source_name,
                    metadata=metadata,
                    import_spec=import_spec,
                )

        dataset = create_dataset_from_pyvista(
            mesh,
            source_path=file_path,
            name=source_name,
            metadata=metadata,
            import_spec=import_spec,
        )
        if isinstance(dataset, SurfaceDataset) and dataset.n_cells == 0:
            return PointSetDataset(
                data=dataset.data,
                source_path=file_path,
                name=source_name,
                metadata=metadata,
                import_spec=import_spec,
            )
        if isinstance(dataset, UnstructuredGridDataset) and getattr(mesh, "n_cells", 0) == 0:
            return PointSetDataset(
                data=pv.PolyData(mesh.points),
                source_path=file_path,
                name=source_name,
                metadata=metadata,
                import_spec=import_spec,
            )
        return dataset

    def _infer_import_spec(self, file_path: str) -> ImportSpec:
        lines = self._read_text_lines(file_path)
        if not lines:
            raise ValueError("文本文件为空。")
        delimiter = self._detect_delimiter(lines[0])
        first_tokens = self._split_line(lines[0], delimiter)
        has_header = not self._tokens_are_numeric(first_tokens)

        headers: list[str]
        if has_header:
            headers = [token.strip() or f"column_{index}" for index, token in enumerate(first_tokens)]
        else:
            headers = [f"column_{index}" for index in range(len(first_tokens))]

        x_column, y_column, z_column = self._infer_axis_columns(headers)
        scalar_columns = [
            header for header in headers if header not in {x_column, y_column, z_column}
        ]

        probe_rows, _ = self._read_numeric_rows(
            file_path,
            ImportSpec(
                file_path=file_path,
                delimiter=delimiter,
                has_header=has_header,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                scalar_columns=scalar_columns,
            ),
            max_rows=5000,
        )
        spec = ImportSpec(
            file_path=file_path,
            delimiter=delimiter,
            has_header=has_header,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            scalar_columns=scalar_columns,
        )
        if probe_rows:
            x = np.array([row[headers.index(x_column)] for row in probe_rows], dtype=float)
            y = np.array([row[headers.index(y_column)] for row in probe_rows], dtype=float)
            z = np.array([row[headers.index(z_column)] for row in probe_rows], dtype=float)
            if self._can_make_image_data(x, y, z, spec):
                spec.is_regular_grid = True
                spec.nx = len(np.unique(x))
                spec.ny = len(np.unique(y))
                spec.nz = len(np.unique(z))
        return spec

    def _read_text_lines(self, file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            return [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith(("#", "//"))
            ]

    def _detect_delimiter(self, line: str) -> str:
        if "," in line:
            return ","
        if "\t" in line:
            return "\t"
        if ";" in line:
            return ";"
        return " "

    def _split_line(self, line: str, delimiter: str) -> list[str]:
        if delimiter == " ":
            return line.split()
        return [token.strip() for token in line.split(delimiter)]

    def _tokens_are_numeric(self, tokens: list[str]) -> bool:
        try:
            for token in tokens:
                float(token)
        except ValueError:
            return False
        return True

    def _infer_axis_columns(self, headers: list[str]) -> tuple[str, str, str]:
        aliases = {
            "x": {"x", "lon", "east", "easting"},
            "y": {"y", "lat", "north", "northing"},
            "z": {"z", "elev", "elevation", "depth"},
        }
        resolved: dict[str, str] = {}
        lowered = {header.lower(): header for header in headers}
        for axis, names in aliases.items():
            for candidate in names:
                if candidate in lowered:
                    resolved[axis] = lowered[candidate]
                    break

        if len(resolved) < 3:
            if len(headers) < 3:
                raise ValueError("至少需要三列数据来表示 x、y、z 坐标。")
            resolved.setdefault("x", headers[0])
            resolved.setdefault("y", headers[1])
            resolved.setdefault("z", headers[2])
        return resolved["x"], resolved["y"], resolved["z"]

    def _read_numeric_rows(
        self,
        file_path: str,
        import_spec: ImportSpec,
        max_rows: int | None = None,
    ) -> tuple[list[list[float]], list[str]]:
        lines = self._read_text_lines(file_path)
        if not lines:
            return [], []

        if import_spec.has_header:
            headers = self._split_line(lines[0], import_spec.delimiter)
            data_lines = lines[1:]
        else:
            first_line_tokens = self._split_line(lines[0], import_spec.delimiter)
            headers = [f"column_{index}" for index in range(len(first_line_tokens))]
            data_lines = lines

        rows: list[list[float]] = []
        for line in data_lines:
            tokens = self._split_line(line, import_spec.delimiter)
            if len(tokens) < len(headers):
                continue
            try:
                rows.append([float(tokens[index]) for index in range(len(headers))])
            except ValueError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
        return rows, headers

    def _can_make_image_data(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: ImportSpec) -> bool:
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        if unique_x.size * unique_y.size * unique_z.size != x.size:
            return False
        for coords in (unique_x, unique_y, unique_z):
            if coords.size <= 2:
                continue
            diffs = np.diff(coords)
            if not np.allclose(diffs, diffs[0]):
                return False
        spec.nx = int(unique_x.size)
        spec.ny = int(unique_y.size)
        spec.nz = int(unique_z.size)
        return True

    def _build_image_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        scalar_columns: list[str],
        column_arrays: dict[str, np.ndarray],
        spec: ImportSpec,
    ) -> pv.ImageData:
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        nx, ny, nz = int(unique_x.size), int(unique_y.size), int(unique_z.size)
        spacing = (
            float(unique_x[1] - unique_x[0]) if nx > 1 else 1.0,
            float(unique_y[1] - unique_y[0]) if ny > 1 else 1.0,
            float(unique_z[1] - unique_z[0]) if nz > 1 else 1.0,
        )
        image = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=(
            float(unique_x.min()),
            float(unique_y.min()),
            float(unique_z.min()),
        ))

        x_index = {value: index for index, value in enumerate(unique_x.tolist())}
        y_index = {value: index for index, value in enumerate(unique_y.tolist())}
        z_index = {value: index for index, value in enumerate(unique_z.tolist())}
        flat_indices = np.array(
            [
                x_index[float(px)] + y_index[float(py)] * nx + z_index[float(pz)] * nx * ny
                for px, py, pz in zip(x, y, z, strict=True)
            ],
            dtype=int,
        )

        for column in scalar_columns:
            values = np.full(nx * ny * nz, np.nan, dtype=float)
            values[flat_indices] = column_arrays[column]
            image.point_data[column] = values
        return image

    def _rectilinear_to_image_if_uniform(self, grid: pv.RectilinearGrid) -> pv.ImageData | None:
        axes = [np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z)]
        for axis in axes:
            if axis.size <= 2:
                continue
            diffs = np.diff(axis)
            if not np.allclose(diffs, diffs[0]):
                return None
        image = pv.ImageData(
            dimensions=grid.dimensions,
            spacing=tuple(float(np.diff(axis)[0]) if axis.size > 1 else 1.0 for axis in axes),
            origin=(float(axes[0][0]), float(axes[1][0]), float(axes[2][0])),
        )
        for name in grid.point_data.keys():
            image.point_data[name] = grid.point_data[name]
        for name in grid.cell_data.keys():
            image.cell_data[name] = grid.cell_data[name]
        return image

    def _structured_to_image_if_uniform(self, grid: pv.StructuredGrid) -> pv.ImageData | None:
        dims = tuple(int(value) for value in grid.dimensions)
        points = np.asarray(grid.points).reshape((dims[2], dims[1], dims[0], 3))
        x_axis = points[0, 0, :, 0]
        y_axis = points[0, :, 0, 1]
        z_axis = points[:, 0, 0, 2]

        if not (
            np.allclose(points[0, 0, :, 1], points[0, 0, 0, 1])
            and np.allclose(points[0, 0, :, 2], points[0, 0, 0, 2])
            and np.allclose(points[0, :, 0, 0], points[0, 0, 0, 0])
            and np.allclose(points[0, :, 0, 2], points[0, 0, 0, 2])
            and np.allclose(points[:, 0, 0, 0], points[0, 0, 0, 0])
            and np.allclose(points[:, 0, 0, 1], points[0, 0, 0, 1])
        ):
            return None
        for axis in (x_axis, y_axis, z_axis):
            if axis.size <= 2:
                continue
            diffs = np.diff(axis)
            if not np.allclose(diffs, diffs[0]):
                return None

        image = pv.ImageData(
            dimensions=dims,
            spacing=(
                float(x_axis[1] - x_axis[0]) if dims[0] > 1 else 1.0,
                float(y_axis[1] - y_axis[0]) if dims[1] > 1 else 1.0,
                float(z_axis[1] - z_axis[0]) if dims[2] > 1 else 1.0,
            ),
            origin=(float(x_axis[0]), float(y_axis[0]), float(z_axis[0])),
        )
        for name in grid.point_data.keys():
            image.point_data[name] = grid.point_data[name]
        for name in grid.cell_data.keys():
            image.cell_data[name] = grid.cell_data[name]
        return image
