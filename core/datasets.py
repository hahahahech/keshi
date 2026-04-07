"""
三维正反演可视化软件的数据集抽象定义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


PREVIEW_CELL_LIMIT = 2_000_000


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class ScalarFieldInfo:
    name: str
    association: str
    components: int
    value_range: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.value_range is not None:
            payload["value_range"] = list(self.value_range)
        return _json_ready(payload)


@dataclass
class ImportSpec:
    file_path: str = ""
    delimiter: str = ","
    has_header: bool = True
    x_column: str = "x"
    y_column: str = "y"
    z_column: str = "z"
    scalar_columns: list[str] = field(default_factory=list)
    units: str = "m"
    is_regular_grid: bool = False
    nx: int | None = None
    ny: int | None = None
    nz: int | None = None
    nodata: float | None = None
    interpolation: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ImportSpec | None":
        if not payload:
            return None
        return cls(**payload)


@dataclass
class DatasetMetadata:
    source_path: str = ""
    source_name: str = ""
    dataset_type: str = "dataset"
    source_schema: dict[str, Any] = field(default_factory=dict)
    units: str = "m"
    nodata: float | None = None
    preview_mode: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


class BaseDataset:
    dataset_kind = "dataset"

    def __init__(
        self,
        data: pv.DataSet,
        source_path: str = "",
        name: str | None = None,
        metadata: DatasetMetadata | None = None,
        import_spec: ImportSpec | None = None,
    ):
        dataset_name = name or (Path(source_path).stem if source_path else self.dataset_kind)
        self.data = self._normalize_data(data)
        self.import_spec = import_spec
        self.metadata = metadata or DatasetMetadata(
            source_path=source_path,
            source_name=dataset_name,
            dataset_type=self.dataset_kind,
            source_schema=(import_spec.to_dict() if import_spec else {}),
        )
        self.metadata.source_path = source_path or self.metadata.source_path
        self.metadata.source_name = dataset_name or self.metadata.source_name
        self.metadata.dataset_type = self.dataset_kind
        self.metadata.source_schema = (
            import_spec.to_dict() if import_spec else self.metadata.source_schema
        )
        self.metadata.preview_mode = bool(
            self.metadata.preview_mode or getattr(self.data, "n_cells", 0) > PREVIEW_CELL_LIMIT
        )
        self.name = dataset_name
        self.scalar_fields = self._extract_scalar_fields(self.data)
        self.active_scalar = self._choose_default_scalar()
        self.value_range = self.get_scalar_range()

    @staticmethod
    def _normalize_data(data: pv.DataSet) -> pv.DataSet:
        if isinstance(data, pv.MultiBlock):
            if data.n_blocks == 0:
                raise ValueError("文件中没有可显示的几何对象。")
            try:
                combined = data.combine(merge_points=False)
            except Exception as exc:
                raise ValueError(f"无法合并多块数据集：{exc}") from exc
            data = combined

        if data is None or getattr(data, "n_points", 0) == 0:
            raise ValueError("数据集中没有可显示的几何对象。")
        return data

    def _extract_scalar_fields(self, data: pv.DataSet) -> dict[str, ScalarFieldInfo]:
        scalar_fields: dict[str, ScalarFieldInfo] = {}
        for association, container in (("point", data.point_data), ("cell", data.cell_data)):
            for name in list(container.keys()):
                array = np.asarray(container[name])
                if array.size == 0:
                    continue
                if array.ndim > 1 and array.shape[1] != 1:
                    continue
                values = array.reshape(-1)
                finite = values[np.isfinite(values)]
                value_range = None
                if finite.size:
                    value_range = (float(finite.min()), float(finite.max()))
                scalar_fields[name] = ScalarFieldInfo(
                    name=name,
                    association=association,
                    components=1 if array.ndim == 1 else int(array.shape[1]),
                    value_range=value_range,
                )
        return scalar_fields

    def _choose_default_scalar(self) -> str | None:
        if self.scalar_fields:
            return next(iter(self.scalar_fields.keys()))
        return None

    @property
    def source_path(self) -> str:
        return self.metadata.source_path

    @property
    def source_schema(self) -> dict[str, Any]:
        return self.metadata.source_schema

    @property
    def bounds(self):
        return self.data.bounds

    @property
    def units(self) -> str:
        return self.metadata.units

    @property
    def nodata(self) -> float | None:
        return self.metadata.nodata

    @property
    def n_points(self) -> int:
        return int(getattr(self.data, "n_points", 0))

    @property
    def n_cells(self) -> int:
        return int(getattr(self.data, "n_cells", 0))

    @property
    def scalar_names(self) -> list[str]:
        return list(self.scalar_fields.keys())

    @property
    def is_point_set(self) -> bool:
        return isinstance(self, PointSetDataset)

    @property
    def is_regular_grid(self) -> bool:
        return isinstance(self, RegularGridDataset)

    def set_active_scalar(self, scalar_name: str | None):
        if scalar_name is not None and scalar_name not in self.scalar_fields:
            raise KeyError(f"未知标量属性：{scalar_name}")
        self.active_scalar = scalar_name
        self.value_range = self.get_scalar_range()

    def get_scalar_info(self, scalar_name: str | None = None) -> ScalarFieldInfo | None:
        target = scalar_name or self.active_scalar
        if target is None:
            return None
        return self.scalar_fields.get(target)

    def get_scalar_association(self, scalar_name: str | None = None) -> str:
        info = self.get_scalar_info(scalar_name)
        return info.association if info else "point"

    def get_scalar_range(self, scalar_name: str | None = None) -> tuple[float, float] | None:
        info = self.get_scalar_info(scalar_name)
        return info.value_range if info else None

    def get_render_data(self, scalar_name: str | None = None) -> tuple[pv.DataSet, str | None]:
        target = scalar_name or self.active_scalar
        return self.data, target

    def get_volume_render_data(self, scalar_name: str | None = None) -> tuple[pv.DataSet, str | None]:
        return self.get_render_data(scalar_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_kind": self.dataset_kind,
            "source_path": self.source_path,
            "metadata": self.metadata.to_dict(),
            "import_spec": self.import_spec.to_dict() if self.import_spec else None,
            "scalar_fields": {name: info.to_dict() for name, info in self.scalar_fields.items()},
            "active_scalar": self.active_scalar,
            "value_range": list(self.value_range) if self.value_range else None,
        }


class RegularGridDataset(BaseDataset):
    dataset_kind = "regular_grid"

    def __init__(
        self,
        data: pv.ImageData,
        source_path: str = "",
        name: str | None = None,
        metadata: DatasetMetadata | None = None,
        import_spec: ImportSpec | None = None,
    ):
        if not isinstance(data, pv.ImageData):
            raise TypeError("RegularGridDataset 需要 pyvista.ImageData。")
        super().__init__(
            data=data,
            source_path=source_path,
            name=name,
            metadata=metadata,
            import_spec=import_spec,
        )

    def get_volume_render_data(self, scalar_name: str | None = None) -> tuple[pv.DataSet, str | None]:
        target = scalar_name or self.active_scalar
        if target is None:
            return self.data, None
        association = self.get_scalar_association(target)
        if association == "cell":
            converted = self.data.cell_data_to_point_data(pass_cell_data=True)
            return converted, target
        return self.data, target


class UnstructuredGridDataset(BaseDataset):
    dataset_kind = "unstructured_grid"


class SurfaceDataset(BaseDataset):
    dataset_kind = "surface"


class PointSetDataset(BaseDataset):
    dataset_kind = "point_set"


class MeshDataset(SurfaceDataset):
    """兼容旧版仅网格工作流的别名类型。"""


def create_dataset_from_pyvista(
    data: pv.DataSet,
    source_path: str = "",
    name: str | None = None,
    metadata: DatasetMetadata | None = None,
    import_spec: ImportSpec | None = None,
) -> BaseDataset:
    if isinstance(data, pv.MultiBlock):
        data = BaseDataset._normalize_data(data)

    if isinstance(data, pv.ImageData):
        return RegularGridDataset(
            data=data,
            source_path=source_path,
            name=name,
            metadata=metadata,
            import_spec=import_spec,
        )

    if isinstance(data, pv.PolyData):
        if getattr(data, "n_cells", 0) == 0:
            return PointSetDataset(
                data=data,
                source_path=source_path,
                name=name,
                metadata=metadata,
                import_spec=import_spec,
            )
        return SurfaceDataset(
            data=data,
            source_path=source_path,
            name=name,
            metadata=metadata,
            import_spec=import_spec,
        )

    return UnstructuredGridDataset(
        data=data,
        source_path=source_path,
        name=name,
        metadata=metadata,
        import_spec=import_spec,
    )
