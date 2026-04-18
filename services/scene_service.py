"""
场景状态与渲染编排的应用服务。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from core.datasets import (
    BaseDataset,
    DatasetMetadata,
    ImportSpec,
    PointSetDataset,
    RegularGridDataset,
    create_dataset_from_pyvista,
)
from core.project import GeologicalProject
from core.scene import DatasetSceneObject, MeshSceneObject
from core.styles import RenderStyle
from rendering.render_manager import RenderManager


class SceneService:
    def __init__(
        self,
        project: GeologicalProject | None = None,
        render_manager: RenderManager | None = None,
    ):
        self.project = project or GeologicalProject()
        self.render_manager = render_manager or RenderManager()

    def set_plotter(self, plotter):
        self.render_manager.set_plotter(plotter)

    def reset_project(self, name: str = "未命名"):
        self.clear_scene()
        self.project = GeologicalProject(name=name)
        self.render_manager = RenderManager(self.render_manager.plotter)

    def clear_scene(self):
        for scene_object in self.project.scene.clear():
            scene_object.cleanup(plotter=self.render_manager.plotter)

    def add_object(self, scene_object, render: bool = True):
        scene_object.attach_render_manager(self.render_manager)
        self.project.scene.add_object(scene_object)
        if render and self.render_manager.plotter is not None:
            self.render_manager.render_object(scene_object)
        return scene_object

    def create_dataset_object(
        self,
        dataset: BaseDataset,
        *,
        name: str | None = None,
        object_type: str = "dataset",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_object_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> DatasetSceneObject:
        object_style = (style or self._default_style_for_dataset(dataset)).normalized()
        return DatasetSceneObject(
            dataset=dataset,
            name=name,
            object_type=object_type,
            style=object_style,
            object_id=object_id,
            metadata=metadata,
            source_object_id=source_object_id,
            parameters=parameters,
        )

    def add_dataset(
        self,
        dataset: BaseDataset,
        *,
        render: bool = True,
        name: str | None = None,
        object_type: str = "dataset",
        style: RenderStyle | None = None,
        object_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_object_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        scene_object = self.create_dataset_object(
            dataset,
            name=name,
            object_type=object_type,
            style=style,
            object_id=object_id,
            metadata=metadata,
            source_object_id=source_object_id,
            parameters=parameters,
        )
        return self.add_object(scene_object, render=render)

    def create_model_object(self, dataset):
        return MeshSceneObject(dataset=dataset, object_type="dataset")

    def add_dataset_as_model(self, dataset, render: bool = True):
        return self.add_dataset(dataset, render=render, object_type="dataset")

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

    def clear_derived_objects(self, source_object_id: str | None = None):
        removable_ids = []
        for scene_object in self.all_objects():
            if scene_object.object_type == "dataset":
                continue
            if source_object_id is None or scene_object.source_object_id == source_object_id:
                removable_ids.append(scene_object.object_id)
        for object_id in removable_ids:
            self.remove_object(object_id)

    def rename_object(self, object_id: str, new_name: str):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            raise KeyError(f"未知场景对象：{object_id}")
        scene_object.name = new_name
        return scene_object

    def set_visibility(self, object_id: str, visible: bool):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            return None
        scene_object.visible = visible
        self.render_manager.apply_style(scene_object)
        return scene_object

    def set_opacity(self, object_id: str, opacity: float):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            return None
        scene_object.opacity = opacity
        if scene_object.render_mode == "volume":
            self.rerender_object(object_id)
        else:
            self.render_manager.apply_style(scene_object)
        return scene_object

    def update_style(self, object_id: str, **style_updates):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            return None

        if "scalar_name" in style_updates:
            scene_object.set_active_scalar(style_updates["scalar_name"])
            style_updates.pop("scalar_name")

        for key, value in style_updates.items():
            if hasattr(scene_object.style, key):
                setattr(scene_object.style, key, value)
        scene_object.style = scene_object.style.normalized()
        if scene_object.style.clim is None:
            scene_object.style.clim = scene_object.dataset.get_scalar_range(scene_object.active_scalar)
        self.rerender_object(object_id)
        return scene_object

    def rerender_object(self, object_id: str):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            return None
        if self.render_manager.plotter is not None:
            self.render_manager.render_object(scene_object)
        return scene_object

    def rerender_all(self):
        for scene_object in self.all_objects():
            self.rerender_object(scene_object.object_id)

    def create_axis_slice(
        self,
        source_object_id: str,
        axis: str,
        position: float,
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        bounds = np.array(scene_object.bounds, dtype=float)
        axis = axis.lower()
        centers = {
            "x": np.array([position, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]),
            "y": np.array([(bounds[0] + bounds[1]) / 2.0, position, (bounds[4] + bounds[5]) / 2.0]),
            "z": np.array([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, position]),
        }
        normals = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        if axis not in normals:
            raise ValueError("坐标轴必须是 x、y、z 之一。")
        result = scene_object.data.slice(normal=normals[axis], origin=centers[axis])
        name = f"{scene_object.name} {axis.upper()} 向切片 {position:.3f}"
        parameters = {"kind": "axis", "axis": axis, "position": float(position)}
        return self._add_derived_object(
            scene_object,
            result,
            object_type="slice",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def create_axis_slice_batch(
        self,
        source_object_id: str,
        axis: str,
        start: float,
        end: float,
        step: float,
        *,
        render: bool = True,
        add_to_scene: bool = True,
    ) -> list[DatasetSceneObject]:
        axis = axis.lower()
        if step <= 0:
            raise ValueError("步长必须大于 0。")

        low = float(min(start, end))
        high = float(max(start, end))
        if np.isclose(low, high):
            return [
                self.create_axis_slice(
                    source_object_id,
                    axis,
                    low,
                    render=render,
                    add_to_scene=add_to_scene,
                )
            ]

        positions = np.arange(low, high + step * 0.5, step, dtype=float)
        if positions.size > 500:
            raise ValueError("批量切片数量过多，请增大步长或缩小范围。")

        results: list[DatasetSceneObject] = []
        for position in positions:
            try:
                results.append(
                    self.create_axis_slice(
                        source_object_id,
                        axis,
                        float(position),
                        render=render,
                        add_to_scene=add_to_scene,
                    )
                )
            except ValueError as exc:
                if str(exc) != "该操作结果为空。":
                    raise
                continue
        if not results:
            raise ValueError("指定范围内未生成任何有效切片，请调整起点、终点或步长。")
        return results

    def create_orthogonal_slice(
        self,
        source_object_id: str,
        x: float,
        y: float,
        z: float,
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        result = scene_object.data.slice_orthogonal(x=x, y=y, z=z)
        name = f"{scene_object.name} 三向切片"
        parameters = {"kind": "orthogonal", "x": float(x), "y": float(y), "z": float(z)}
        return self._add_derived_object(
            scene_object,
            result,
            object_type="slice",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def create_plane_slice(
        self,
        source_object_id: str,
        origin: tuple[float, float, float],
        normal: tuple[float, float, float],
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        result = scene_object.data.slice(normal=normal, origin=origin)
        name = f"{scene_object.name} 平面切片"
        parameters = {
            "kind": "plane",
            "origin": [float(value) for value in origin],
            "normal": [float(value) for value in normal],
        }
        return self._add_derived_object(
            scene_object,
            result,
            object_type="slice",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def create_polyline_section(
        self,
        source_object_id: str,
        polyline_points,
        *,
        top_z: float,
        bottom_z: float,
        line_step: float = 25.0,
        vertical_samples: int = 20,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        if isinstance(scene_object.dataset, PointSetDataset):
            raise ValueError("点集数据请先转规则网格后再生成折线剖面。")
        if line_step <= 0:
            raise ValueError("沿线步长必须大于 0。")
        if int(vertical_samples) < 2:
            raise ValueError("垂向采样层数至少为 2。")

        points = self._normalize_polyline_points(polyline_points)
        samples, distances = self._resample_polyline_points(points, float(line_step))
        if np.isclose(top_z, bottom_z):
            raise ValueError("顶部 Z 和底部 Z 不能相同。")

        vertical_levels = np.linspace(float(top_z), float(bottom_z), int(vertical_samples))
        x_coords = np.repeat(samples[:, 0][None, :], len(vertical_levels), axis=0)
        y_coords = np.repeat(samples[:, 1][None, :], len(vertical_levels), axis=0)
        z_coords = np.repeat(vertical_levels[:, None], len(samples), axis=1)
        fence = pv.StructuredGrid(x_coords, y_coords, z_coords)
        fence.point_data["profile_distance"] = np.repeat(distances[None, :], len(vertical_levels), axis=0).ravel(order="F")
        fence.point_data["elevation"] = z_coords.ravel(order="F")

        sampled = fence.sample(scene_object.data)
        result = sampled.extract_surface()
        name = f"{scene_object.name} 折线剖面"
        parameters = {
            "kind": "polyline",
            "points": [[float(value) for value in point] for point in points.tolist()],
            "top_z": float(top_z),
            "bottom_z": float(bottom_z),
            "line_step": float(line_step),
            "vertical_samples": int(vertical_samples),
        }
        return self._add_derived_object(
            scene_object,
            result,
            object_type="section",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def create_clip_box(
        self,
        source_object_id: str,
        bounds: tuple[float, float, float, float, float, float],
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        if isinstance(scene_object.dataset, RegularGridDataset):
            result = self._clip_regular_grid_box(scene_object.data, bounds)
        else:
            result = scene_object.data.clip_box(bounds=bounds, invert=False)
        name = f"{scene_object.name} 裁剪结果"
        parameters = {"bounds": [float(value) for value in bounds]}
        return self._add_derived_object(
            scene_object,
            result,
            object_type="clip",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def create_isosurface(
        self,
        source_object_id: str,
        value: float,
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        scalar_name = scene_object.active_scalar
        if scalar_name is None:
            raise ValueError("生成等值面前必须先选择一个标量属性。")
        result = scene_object.data.contour(isosurfaces=[float(value)], scalars=scalar_name)
        name = f"{scene_object.name} 等值面 {value:.3g}"
        parameters = {"value": float(value), "scalar_name": scalar_name}
        return self._add_derived_object(
            scene_object,
            result,
            object_type="isosurface",
            name=name,
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
            parameters=parameters,
        )

    def interpolate_point_dataset_to_grid(
        self,
        source_object_id: str,
        dimensions: tuple[int, int, int] = (30, 30, 30),
        power: float = 2.0,
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        if not isinstance(scene_object.dataset, PointSetDataset):
            raise ValueError("仅点集数据支持插值生成规则格网。")
        scalar_name = scene_object.active_scalar
        if scalar_name is None:
            raise ValueError("当前点集没有可用于插值的标量属性。")

        nx, ny, nz = [max(2, int(value)) for value in dimensions]
        bounds = np.array(scene_object.bounds, dtype=float)
        x_axis = np.linspace(bounds[0], bounds[1], nx)
        y_axis = np.linspace(bounds[2], bounds[3], ny)
        z_axis = np.linspace(bounds[4], bounds[5], nz)
        gx, gy, gz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
        target_points = np.column_stack(
            [gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")]
        )
        source_points = np.asarray(scene_object.data.points, dtype=float)
        source_values = np.asarray(scene_object.data.point_data[scalar_name], dtype=float).reshape(-1)
        interpolated = self._idw_interpolate(
            source_points,
            source_values,
            target_points,
            power=float(power),
        )

        image = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(
                float(x_axis[1] - x_axis[0]) if nx > 1 else 1.0,
                float(y_axis[1] - y_axis[0]) if ny > 1 else 1.0,
                float(z_axis[1] - z_axis[0]) if nz > 1 else 1.0,
            ),
            origin=(float(x_axis[0]), float(y_axis[0]), float(z_axis[0])),
        )
        image.point_data[scalar_name] = interpolated
        dataset = RegularGridDataset(
            data=image,
            source_path=scene_object.dataset.source_path,
            name=f"{scene_object.name} 反距离加权格网",
            metadata=DatasetMetadata(
                source_path=scene_object.dataset.source_path,
                source_name=f"{scene_object.name} 反距离加权格网",
                dataset_type="regular_grid",
                source_schema={"derived_from": source_object_id, "operation": "idw"},
                units=scene_object.dataset.units,
                nodata=scene_object.dataset.nodata,
            ),
            import_spec=scene_object.dataset.import_spec,
        )
        style = self._copy_style(scene_object.style)
        style.render_mode = "volume"
        style.scalar_name = scalar_name
        style.clim = dataset.get_scalar_range(scalar_name)
        parameters = {
            "dimensions": [nx, ny, nz],
            "power": float(power),
            "scalar_name": scalar_name,
        }
        if add_to_scene:
            return self.add_dataset(
                dataset,
                render=render,
                name=dataset.name,
                object_type="dataset",
                style=style,
                object_id=object_id,
                metadata={"derived": True},
                source_object_id=source_object_id,
                parameters=parameters,
            )
        return self.create_dataset_object(
            dataset,
            name=dataset.name,
            object_type="dataset",
            style=style,
            object_id=object_id,
            metadata={"derived": True},
            source_object_id=source_object_id,
            parameters=parameters,
        )

    def serialize_scene(self) -> list[dict[str, Any]]:
        return [scene_object.to_dict() for scene_object in self.all_objects()]

    def load_from_payload(self, payload: dict[str, Any], import_service) -> list[DatasetSceneObject]:
        self.clear_scene()
        self.project.name = payload.get("name", "未命名")
        self.project.metadata = payload.get("metadata", {})
        restored: list[DatasetSceneObject] = []
        definitions = payload.get("objects", [])
        ordered_objects: dict[str, DatasetSceneObject] = {}

        for definition in definitions:
            if definition.get("object_type") != "dataset":
                continue
            dataset_info = definition.get("dataset", {})
            file_path = dataset_info.get("source_path")
            if not file_path:
                continue
            import_spec = dataset_info.get("import_spec")
            dataset = import_service.load_dataset(
                file_path,
                import_spec=ImportSpec.from_dict(import_spec),
            )
            scene_object = self.add_dataset(
                dataset,
                render=False,
                name=definition.get("name"),
                object_type="dataset",
                style=RenderStyle.from_dict(definition.get("style")),
                object_id=definition.get("object_id"),
                metadata=definition.get("metadata"),
                source_object_id=definition.get("source_object_id"),
                parameters=definition.get("parameters"),
            )
            ordered_objects[scene_object.object_id] = scene_object
            restored.append(scene_object)

        for definition in definitions:
            object_type = definition.get("object_type")
            if object_type == "dataset":
                continue
            source_object_id = definition.get("source_object_id")
            if source_object_id not in ordered_objects:
                continue
            parameters = definition.get("parameters", {})
            if object_type == "slice":
                kind = parameters.get("kind")
                if kind == "axis":
                    scene_object = self.create_axis_slice(
                        source_object_id,
                        parameters["axis"],
                        parameters["position"],
                        render=False,
                        object_id=definition.get("object_id"),
                    )
                elif kind == "orthogonal":
                    scene_object = self.create_orthogonal_slice(
                        source_object_id,
                        parameters["x"],
                        parameters["y"],
                        parameters["z"],
                        render=False,
                        object_id=definition.get("object_id"),
                    )
                else:
                    scene_object = self.create_plane_slice(
                        source_object_id,
                        tuple(parameters["origin"]),
                        tuple(parameters["normal"]),
                        render=False,
                        object_id=definition.get("object_id"),
                    )
            elif object_type == "clip":
                scene_object = self.create_clip_box(
                    source_object_id,
                    tuple(parameters["bounds"]),
                    render=False,
                    object_id=definition.get("object_id"),
                )
            elif object_type == "section":
                scene_object = self.create_polyline_section(
                    source_object_id,
                    parameters["points"],
                    top_z=parameters["top_z"],
                    bottom_z=parameters["bottom_z"],
                    line_step=parameters["line_step"],
                    vertical_samples=parameters["vertical_samples"],
                    render=False,
                    object_id=definition.get("object_id"),
                )
            elif object_type == "isosurface":
                scene_object = self.create_isosurface(
                    source_object_id,
                    parameters["value"],
                    render=False,
                    object_id=definition.get("object_id"),
                )
            else:
                continue
            scene_object.name = definition.get("name", scene_object.name)
            scene_object.style = RenderStyle.from_dict(definition.get("style"))
            self.rerender_object(scene_object.object_id)
            ordered_objects[scene_object.object_id] = scene_object
            restored.append(scene_object)

        self.project.camera_state = payload.get("camera_state", {})
        self.rerender_all()
        return restored

    def _default_style_for_dataset(self, dataset: BaseDataset) -> RenderStyle:
        render_mode = "surface"
        if isinstance(dataset, RegularGridDataset):
            render_mode = "volume"
        if isinstance(dataset, PointSetDataset):
            render_mode = "points"
        return RenderStyle(
            opacity=1.0,
            render_mode=render_mode,
            scalar_name=dataset.active_scalar,
            clim=dataset.get_scalar_range(),
            show_scalar_bar=dataset.active_scalar is not None,
        ).normalized()

    def _copy_style(self, style: RenderStyle) -> RenderStyle:
        return RenderStyle.from_dict(style.to_dict())

    def _require_object(self, object_id: str):
        scene_object = self.get_object(object_id)
        if scene_object is None:
            raise KeyError(f"未知场景对象：{object_id}")
        return scene_object

    def _normalize_polyline_points(self, polyline_points) -> np.ndarray:
        points = np.asarray(polyline_points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            raise ValueError("折线至少需要两个点。")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0], dtype=float)])
        if points.shape[1] != 3:
            raise ValueError("折线点必须是二维或三维坐标。")
        return points

    def _resample_polyline_points(self, points: np.ndarray, line_step: float) -> tuple[np.ndarray, np.ndarray]:
        xy_points = points[:, :2]
        segment_lengths = np.linalg.norm(np.diff(xy_points, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length <= 1e-9:
            raise ValueError("折线长度过短，无法生成剖面。")

        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        sample_distances = np.arange(0.0, total_length + line_step * 0.5, line_step, dtype=float)
        if sample_distances.size == 0 or sample_distances[-1] < total_length:
            sample_distances = np.append(sample_distances, total_length)

        samples = []
        for distance in sample_distances:
            index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(segment_lengths) - 1)
            segment_length = max(float(segment_lengths[index]), 1e-9)
            ratio = (float(distance) - float(cumulative[index])) / segment_length
            point = points[index] + (points[index + 1] - points[index]) * ratio
            samples.append(point)
        return np.asarray(samples, dtype=float), sample_distances

    def _clip_regular_grid_box(
        self,
        data: pv.ImageData,
        bounds: tuple[float, float, float, float, float, float],
    ) -> pv.ImageData:
        bounds_array = np.asarray(bounds, dtype=float).reshape(-1)
        if bounds_array.size != 6:
            raise ValueError("裁剪范围必须包含 6 个数值。")

        voi: list[int] = []
        dimensions = np.asarray(data.dimensions, dtype=int)
        origin = np.asarray(data.origin, dtype=float)
        spacing = np.asarray(data.spacing, dtype=float)

        for axis in range(3):
            low = float(min(bounds_array[axis * 2], bounds_array[axis * 2 + 1]))
            high = float(max(bounds_array[axis * 2], bounds_array[axis * 2 + 1]))
            coords = origin[axis] + np.arange(int(dimensions[axis]), dtype=float) * spacing[axis]
            start = int(np.searchsorted(coords, low, side="left"))
            end = int(np.searchsorted(coords, high, side="right") - 1)
            start = max(0, min(start, int(dimensions[axis]) - 1))
            end = max(0, min(end, int(dimensions[axis]) - 1))
            if start > end:
                raise ValueError("该操作结果为空。")
            voi.extend([start, end])

        return data.extract_subset(tuple(voi), rebase_coordinates=False)

    def _result_has_geometry(self, result: pv.DataSet) -> bool:
        if isinstance(result, pv.MultiBlock):
            for block in result:
                if block is not None and getattr(block, "n_points", 0) > 0:
                    return True
            return False
        return getattr(result, "n_points", 0) > 0

    def _add_derived_object(
        self,
        source_object,
        result: pv.DataSet,
        *,
        object_type: str,
        name: str,
        parameters: dict[str, Any],
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        if not self._result_has_geometry(result):
            raise ValueError("该操作结果为空。")
        dataset = create_dataset_from_pyvista(
            result,
            source_path=source_object.dataset.source_path,
            name=name,
            metadata=DatasetMetadata(
                source_path=source_object.dataset.source_path,
                source_name=name,
                dataset_type=object_type,
                source_schema={"derived_from": source_object.object_id, "operation": object_type},
                units=source_object.dataset.units,
                nodata=source_object.dataset.nodata,
            ),
            import_spec=source_object.dataset.import_spec,
        )
        style = self._copy_style(source_object.style)
        if object_type == "clip":
            style.render_mode = source_object.style.render_mode
        else:
            style.render_mode = "surface"
        style.scalar_name = source_object.active_scalar
        style.clim = source_object.style.clim or dataset.get_scalar_range(style.scalar_name)
        if add_to_scene:
            return self.add_dataset(
                dataset,
                render=render,
                name=name,
                object_type=object_type,
                style=style,
                object_id=object_id,
                source_object_id=source_object.object_id,
                parameters=parameters,
            )
        return self.create_dataset_object(
            dataset,
            name=name,
            object_type=object_type,
            style=style,
            object_id=object_id,
            source_object_id=source_object.object_id,
            parameters=parameters,
        )

    def _idw_interpolate(
        self,
        source_points: np.ndarray,
        source_values: np.ndarray,
        target_points: np.ndarray,
        *,
        power: float = 2.0,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        result = np.empty(target_points.shape[0], dtype=float)
        epsilon = 1e-12
        for start in range(0, target_points.shape[0], chunk_size):
            end = min(start + chunk_size, target_points.shape[0])
            chunk = target_points[start:end]
            distances = np.linalg.norm(chunk[:, None, :] - source_points[None, :, :], axis=2)
            exact_matches = distances <= epsilon
            if np.any(exact_matches):
                for row_index, mask in enumerate(exact_matches):
                    if np.any(mask):
                        result[start + row_index] = float(source_values[np.argmax(mask)])
                unresolved = ~np.any(exact_matches, axis=1)
            else:
                unresolved = np.ones(chunk.shape[0], dtype=bool)

            if np.any(unresolved):
                unresolved_distances = distances[unresolved]
                weights = 1.0 / np.maximum(unresolved_distances, epsilon) ** power
                weight_sums = weights.sum(axis=1)
                chunk_result = result[start:end]
                chunk_result[unresolved] = (weights @ source_values) / weight_sums
                result[start:end] = chunk_result
        return result
