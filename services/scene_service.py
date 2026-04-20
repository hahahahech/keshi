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
        draw_plane: str = "xoy",
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
        plane = self._normalize_draw_plane(draw_plane)
        line_axes, section_axis = self._draw_plane_axes(plane)
        samples, distances = self._resample_polyline_points_by_axes(points, line_axes, float(line_step))
        if np.isclose(top_z, bottom_z):
            raise ValueError("剖面上界和下界不能相同。")

        direction_2d = self._principal_direction_on_axes(points, line_axes)
        line_direction = np.zeros(3, dtype=float)
        line_direction[line_axes[0]] = direction_2d[0]
        line_direction[line_axes[1]] = direction_2d[1]
        section_direction = np.zeros(3, dtype=float)
        section_direction[section_axis] = 1.0
        profile_normal = np.cross(line_direction, section_direction)
        normal_norm = float(np.linalg.norm(profile_normal))
        if normal_norm <= 1e-12:
            raise ValueError("折线方向无效，无法生成剖面。")
        profile_normal = profile_normal / normal_norm

        section_levels = np.linspace(float(top_z), float(bottom_z), int(vertical_samples))
        x_coords = np.repeat(samples[:, 0][None, :], len(section_levels), axis=0)
        y_coords = np.repeat(samples[:, 1][None, :], len(section_levels), axis=0)
        z_coords = np.repeat(samples[:, 2][None, :], len(section_levels), axis=0)
        level_grid = np.repeat(section_levels[:, None], len(samples), axis=1)
        if section_axis == 0:
            x_coords = level_grid
        elif section_axis == 1:
            y_coords = level_grid
        else:
            z_coords = level_grid
        fence = pv.StructuredGrid(x_coords, y_coords, z_coords)
        fence.point_data["profile_distance"] = np.repeat(distances[None, :], len(section_levels), axis=0).ravel(order="F")
        fence.point_data["elevation"] = level_grid.ravel(order="F")

        sampled = fence.sample(scene_object.data)
        result = sampled.extract_surface()
        name = f"{scene_object.name} 折线剖面"
        parameters = {
            "kind": "polyline",
            "line_mode": "polyline_fence",
            "points": [[float(value) for value in point] for point in points.tolist()],
            "top_z": float(top_z),
            "bottom_z": float(bottom_z),
            "section_max": float(top_z),
            "section_min": float(bottom_z),
            "draw_plane": plane,
            "line_step": float(line_step),
            "vertical_samples": int(vertical_samples),
            "normal": [float(value) for value in profile_normal.tolist()],
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

    def create_grid_index_clip(
        self,
        source_object_id: str,
        index_bounds: tuple[int, int, int, int, int, int],
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        if not isinstance(scene_object.dataset, RegularGridDataset):
            raise ValueError("按格点索引裁剪仅支持规则体数据。")
        result = self._clip_regular_grid_index_range(scene_object.data, index_bounds)
        name = f"{scene_object.name} 格点裁剪结果"
        parameters = {"index_bounds": [int(value) for value in index_bounds]}
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

    def create_polyline_plane_slice(
        self,
        source_object_id: str,
        polyline_points,
        *,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        points = self._normalize_polyline_points(polyline_points)
        if points.shape[0] < 2:
            raise ValueError("折线至少需要两个点。")

        direction_xy = self._principal_direction_xy(points)
        line_direction = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        normal = np.cross(line_direction, up)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("折线方向无效，无法生成平面切片。")
        normal = normal / normal_norm

        origin = np.asarray(points, dtype=float).mean(axis=0)
        result = scene_object.data.slice(
            normal=tuple(float(v) for v in normal),
            origin=tuple(float(v) for v in origin),
        )
        name = f"{scene_object.name} 折线平面切片"
        parameters = {
            "kind": "plane",
            "origin": [float(v) for v in origin.tolist()],
            "normal": [float(v) for v in normal.tolist()],
            "line_mode": "polyline_plane",
            "line_points": [[float(v) for v in point] for point in points.tolist()],
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

    def move_slice(
        self,
        slice_object_id: str,
        offset: float,
        *,
        component_axis: str | None = None,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        slice_object = self._require_object(slice_object_id)
        kind = str((slice_object.parameters or {}).get("kind") or "").lower()
        if kind == "polyline":
            return self._move_polyline_slice(
                slice_object_id,
                offset=float(offset),
                render=render,
                add_to_scene=add_to_scene,
                object_id=object_id,
            )
        source_object, origin, normal = self._resolve_slice_plane(
            slice_object_id,
            component_axis=component_axis,
        )
        shifted_origin = origin + float(offset) * normal
        return self.create_plane_slice(
            source_object.object_id,
            tuple(float(value) for value in shifted_origin),
            tuple(float(value) for value in normal),
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
        )

    def tilt_slice(
        self,
        slice_object_id: str,
        angle_deg: float,
        *,
        tilt_axis: str | tuple[float, float, float] | list[float],
        component_axis: str | None = None,
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        slice_object = self._require_object(slice_object_id)
        kind = str((slice_object.parameters or {}).get("kind") or "").lower()
        if kind == "polyline":
            return self._tilt_polyline_slice(
                slice_object_id,
                angle_deg=float(angle_deg),
                tilt_axis=tilt_axis,
                render=render,
                add_to_scene=add_to_scene,
                object_id=object_id,
            )
        source_object, origin, normal = self._resolve_slice_plane(
            slice_object_id,
            component_axis=component_axis,
        )
        axis_vector = self._parse_tilt_axis(tilt_axis)
        if abs(float(np.dot(axis_vector, normal))) > 0.999:
            raise ValueError("倾斜轴不能与切片法向平行。")
        rotated_normal = self._rotate_vector(normal, axis_vector, np.deg2rad(float(angle_deg)))
        norm = float(np.linalg.norm(rotated_normal))
        if norm <= 1e-12:
            raise ValueError("倾斜后的切片法向无效。")
        rotated_normal = rotated_normal / norm
        return self.create_plane_slice(
            source_object.object_id,
            tuple(float(value) for value in origin),
            tuple(float(value) for value in rotated_normal),
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
        )

    def create_mask_clip_from_polyline(
        self,
        source_object_id: str,
        polyline_points,
        *,
        draw_plane: str = "xoy",
        render: bool = True,
        add_to_scene: bool = True,
        object_id: str | None = None,
    ):
        scene_object = self._require_object(source_object_id)
        if not isinstance(scene_object.dataset, RegularGridDataset):
            raise ValueError("掩膜裁剪目前仅支持规则体数据。")
        points = self._normalize_polyline_points(polyline_points)
        if points.shape[0] < 3:
            raise ValueError("掩膜边界至少需要三个点。")
        plane = self._normalize_draw_plane(draw_plane)
        result = self._mask_regular_grid_with_polygon(scene_object.data, points, draw_plane=plane)
        name = f"{scene_object.name} 掩膜裁剪结果"
        parameters = {
            "mask_kind": f"polyline_{plane}",
            "draw_plane": plane,
            "points": [[float(value) for value in row] for row in points.tolist()],
        }
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

    def build_well_trajectory_points(
        self,
        *,
        well_object_id: str | None = None,
        trajectory_points=None,
        well_index: int | None = None,
        well_name: str | None = None,
    ) -> np.ndarray:
        if trajectory_points is not None:
            return self._normalize_polyline_points(trajectory_points)

        if not well_object_id:
            raise ValueError("请提供井数据对象 ID 或显式井轨迹点。")
        well_object = self._require_object(well_object_id)
        data = well_object.data
        points = np.asarray(data.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("井轨迹点不足，无法生成井筒。")

        selected_well_index = self._resolve_well_index(well_object, well_index=well_index, well_name=well_name)
        well_indices = self._extract_point_data_array(data, "well_index")
        depth_values = self._extract_point_data_array(data, "depth")

        if selected_well_index is not None and well_indices is not None:
            mask = well_indices == int(selected_well_index)
            selected_indices = np.where(mask)[0]
        else:
            selected_indices = np.arange(points.shape[0], dtype=int)

        if selected_indices.size < 2:
            raise ValueError("选定井的有效轨迹点不足两个。")

        polyline_ids = self._extract_well_polyline_ids(
            data,
            selected_indices=selected_indices,
            well_indices=well_indices,
            selected_well_index=selected_well_index,
        )
        if polyline_ids is not None and polyline_ids.size >= 2:
            return points[polyline_ids]

        selected_points = points[selected_indices]
        selected_depth = depth_values[selected_indices] if depth_values is not None else None
        return self._sort_well_points(selected_points, selected_depth)

    def create_drillhole_mapping(
        self,
        source_object_id: str,
        *,
        well_object_id: str | None = None,
        trajectory_points=None,
        overlay_object_ids: list[str] | None = None,
        well_index: int | None = None,
        well_name: str | None = None,
        radius: float = 25.0,
        tube_sides: int = 16,
        render: bool = True,
        add_to_scene: bool = True,
    ) -> list[DatasetSceneObject]:
        source_object = self._require_object(source_object_id)
        if radius <= 0:
            raise ValueError("井筒半径必须大于 0。")
        if int(tube_sides) < 6:
            raise ValueError("井筒边数至少为 6。")

        trajectory = self.build_well_trajectory_points(
            well_object_id=well_object_id,
            trajectory_points=trajectory_points,
            well_index=well_index,
            well_name=well_name,
        )
        trajectory_line = self._build_polyline_from_points(trajectory)
        tube = trajectory_line.tube(
            radius=float(radius),
            n_sides=int(tube_sides),
            capping=True,
        ).triangulate()
        tube_display_data, tube_scalar_name, tube_scalar_range = self._build_drillhole_tube_display_data(
            tube,
            source_object,
        )

        result_objects: list[DatasetSceneObject] = []
        drilled_data = self._apply_drillhole_mask(source_object.data, tube)
        drilled_object = self._add_derived_object(
            source_object,
            drilled_data,
            object_type="clip",
            name=f"{source_object.name} 钻孔后",
            render=render,
            add_to_scene=add_to_scene,
            parameters={
                "clip_kind": "drillhole",
                "radius": float(radius),
                "tube_sides": int(tube_sides),
                "well_object_id": str(well_object_id or ""),
            },
        )
        result_objects.append(drilled_object)

        trajectory_object = self._add_helper_object(
            trajectory_line,
            name="井轨迹",
            color=(1.0, 0.82, 0.2),
            render_mode="wireframe",
            source_object_id=well_object_id or source_object_id,
            parameters={
                "kind": "well_trajectory",
                "points": [[float(v) for v in row] for row in trajectory.tolist()],
            },
            render=render,
            add_to_scene=add_to_scene,
        )
        result_objects.append(trajectory_object)

        tube_object = self._add_helper_object(
            tube_display_data,
            name="井筒",
            color=(0.95, 0.2, 0.2),
            render_mode="surface",
            source_object_id=well_object_id or source_object_id,
            parameters={
                "kind": "well_tube",
                "radius": float(radius),
                "tube_sides": int(tube_sides),
            },
            scalar_name=tube_scalar_name,
            clim=tube_scalar_range,
            colormap=source_object.style.colormap,
            render=render,
            add_to_scene=add_to_scene,
        )
        result_objects.append(tube_object)

        overlay_ids = list(overlay_object_ids or [])
        for overlay_id in overlay_ids:
            if overlay_id in {source_object_id, well_object_id}:
                continue
            overlay_object = self.get_object(overlay_id)
            if overlay_object is None or overlay_object.dataset is None:
                continue
            clipped_overlay_data = self._apply_drillhole_mask(overlay_object.data, tube)
            if not self._result_has_geometry(clipped_overlay_data):
                continue
            overlay_result = self._add_derived_object(
                overlay_object,
                clipped_overlay_data,
                object_type="clip",
                name=f"{overlay_object.name} 钻孔同步裁剪",
                render=render,
                add_to_scene=add_to_scene,
                parameters={
                    "clip_kind": "drillhole_sync",
                    "radius": float(radius),
                    "tube_sides": int(tube_sides),
                    "source_volume_id": source_object_id,
                    "well_object_id": str(well_object_id or ""),
                },
            )
            result_objects.append(overlay_result)
        return result_objects

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
                elif kind == "polyline":
                    scene_object = self.create_polyline_section(
                        source_object_id,
                        parameters["points"],
                        top_z=parameters["top_z"],
                        bottom_z=parameters["bottom_z"],
                        draw_plane=parameters.get("draw_plane", "xoy"),
                        line_step=parameters["line_step"],
                        vertical_samples=parameters["vertical_samples"],
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
                if str(parameters.get("mask_kind", "")).startswith("polyline_") and "points" in parameters:
                    scene_object = self.create_mask_clip_from_polyline(
                        source_object_id,
                        parameters["points"],
                        draw_plane=parameters.get("draw_plane", str(parameters.get("mask_kind", "polyline_xoy")).replace("polyline_", "")),
                        render=False,
                        object_id=definition.get("object_id"),
                    )
                elif "index_bounds" in parameters:
                    scene_object = self.create_grid_index_clip(
                        source_object_id,
                        tuple(parameters["index_bounds"]),
                        render=False,
                        object_id=definition.get("object_id"),
                    )
                else:
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
                    draw_plane=parameters.get("draw_plane", "xoy"),
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

        self.project.camera_state = payload.get("camera_state") or payload.get("view_state", {}).get("camera_state", {})
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

    def _resolve_slice_plane(
        self,
        slice_object_id: str,
        *,
        component_axis: str | None = None,
    ) -> tuple[DatasetSceneObject, np.ndarray, np.ndarray]:
        slice_object = self._require_object(slice_object_id)
        params = dict(slice_object.parameters or {})
        kind = str(params.get("kind") or "plane").lower()
        is_slice_like = str(getattr(slice_object, "object_type", "") or "").lower() == "slice" or kind in {
            "axis",
            "orthogonal",
            "plane",
        }
        if not is_slice_like:
            raise ValueError("仅切片对象支持移动或倾斜。")

        source_object = self._resolve_slice_source_object(slice_object)
        source_bounds = np.asarray(source_object.bounds, dtype=float)

        if kind == "axis":
            axis_name = str(params.get("axis") or "").lower()
            if axis_name not in {"x", "y", "z"}:
                raise ValueError("切片轴向参数无效。")
            position = float(params.get("position"))
            origin = self._axis_origin_from_bounds(source_bounds, axis_name, position)
            normal = self._axis_vector(axis_name)
            return source_object, origin, normal

        if kind == "orthogonal":
            axis_name = str(component_axis or "").lower()
            if axis_name not in {"x", "y", "z"}:
                raise ValueError("三向切片请先指定要操作的轴向（X/Y/Z）。")
            if axis_name not in params:
                raise ValueError("三向切片参数缺少目标轴向坐标。")
            position = float(params[axis_name])
            origin = self._axis_origin_from_bounds(source_bounds, axis_name, position)
            normal = self._axis_vector(axis_name)
            return source_object, origin, normal

        if "origin" not in params or "normal" not in params:
            raise ValueError("切片参数缺少平面原点或法向。")
        origin = np.asarray(params["origin"], dtype=float).reshape(-1)
        normal = np.asarray(params["normal"], dtype=float).reshape(-1)
        if origin.size != 3 or normal.size != 3:
            raise ValueError("切片平面参数格式无效。")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("切片法向量无效。")
        return source_object, origin, normal / normal_norm

    def _resolve_slice_source_object(self, slice_object):
        source_id = slice_object.source_object_id or slice_object.object_id
        source_object = self._require_object(source_id)
        while source_object.object_type == "slice" and source_object.source_object_id:
            source_object = self._require_object(source_object.source_object_id)
        return source_object

    def _axis_origin_from_bounds(
        self,
        bounds: np.ndarray,
        axis_name: str,
        position: float,
    ) -> np.ndarray:
        origin = np.array(
            [
                float((bounds[0] + bounds[1]) / 2.0),
                float((bounds[2] + bounds[3]) / 2.0),
                float((bounds[4] + bounds[5]) / 2.0),
            ],
            dtype=float,
        )
        axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
        origin[axis_index] = float(position)
        return origin

    def _axis_vector(self, axis_name: str) -> np.ndarray:
        mapping = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        vector = mapping.get(axis_name)
        if vector is None:
            raise ValueError("切片轴向必须是 X/Y/Z。")
        return vector

    def _parse_tilt_axis(self, tilt_axis: str | tuple[float, float, float] | list[float]) -> np.ndarray:
        if isinstance(tilt_axis, str):
            axis_name = tilt_axis.strip().lower()
            if axis_name not in {"x", "y", "z"}:
                raise ValueError("倾斜轴必须是 X/Y/Z。")
            return self._axis_vector(axis_name)

        vector = np.asarray(tilt_axis, dtype=float).reshape(-1)
        if vector.size != 3:
            raise ValueError("倾斜轴向量必须包含 3 个数值。")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("倾斜轴向量无效。")
        return vector / norm

    def _normalize_draw_plane(self, draw_plane: str | None) -> str:
        plane = str(draw_plane or "xoy").strip().lower()
        if plane not in {"xoy", "xoz", "yoz"}:
            raise ValueError("绘制平面必须是 xoy、xoz 或 yoz。")
        return plane

    def _draw_plane_axes(self, draw_plane: str) -> tuple[tuple[int, int], int]:
        plane = self._normalize_draw_plane(draw_plane)
        mapping = {
            "xoy": ((0, 1), 2),
            "xoz": ((0, 2), 1),
            "yoz": ((1, 2), 0),
        }
        return mapping[plane]

    def _principal_direction_on_axes(self, points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
        coords = np.asarray(points[:, list(axes)], dtype=float)
        centered = coords - coords.mean(axis=0, keepdims=True)
        norm_centered = float(np.linalg.norm(centered))
        if norm_centered <= 1e-12:
            raise ValueError("折线点重合，无法生成剖面。")

        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values.size > 0 and float(singular_values[0]) > 1e-12:
            direction = np.asarray(vh[0], dtype=float)
        else:
            direction = np.asarray(coords[-1] - coords[0], dtype=float)

        norm_direction = float(np.linalg.norm(direction))
        if norm_direction <= 1e-12:
            raise ValueError("折线方向无效，无法生成剖面。")
        return direction / norm_direction

    def _resample_polyline_points_by_axes(
        self,
        points: np.ndarray,
        axes: tuple[int, int],
        line_step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(points[:, list(axes)], dtype=float)
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
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

    def _move_polyline_slice(
        self,
        slice_object_id: str,
        *,
        offset: float,
        render: bool,
        add_to_scene: bool,
        object_id: str | None,
    ):
        slice_object = self._require_object(slice_object_id)
        params = dict(slice_object.parameters or {})
        points = self._normalize_polyline_points(params.get("points", []))
        draw_plane = self._normalize_draw_plane(params.get("draw_plane", "xoy"))
        line_axes, section_axis = self._draw_plane_axes(draw_plane)
        direction_2d = self._principal_direction_on_axes(points, line_axes)
        line_direction = np.zeros(3, dtype=float)
        line_direction[line_axes[0]] = direction_2d[0]
        line_direction[line_axes[1]] = direction_2d[1]
        section_direction = np.zeros(3, dtype=float)
        section_direction[section_axis] = 1.0
        normal = np.cross(line_direction, section_direction)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("折线方向无效，无法平移。")
        normal = normal / norm
        shifted_points = points + float(offset) * normal

        section_max = float(params.get("section_max", params.get("top_z")))
        section_min = float(params.get("section_min", params.get("bottom_z")))
        source_object = self._resolve_slice_source_object(slice_object)
        return self.create_polyline_section(
            source_object.object_id,
            shifted_points,
            top_z=section_max,
            bottom_z=section_min,
            draw_plane=draw_plane,
            line_step=float(params["line_step"]),
            vertical_samples=int(params["vertical_samples"]),
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
        )

    def _tilt_polyline_slice(
        self,
        slice_object_id: str,
        *,
        angle_deg: float,
        tilt_axis: str | tuple[float, float, float] | list[float],
        render: bool,
        add_to_scene: bool,
        object_id: str | None,
    ):
        axis_vector = self._parse_tilt_axis(tilt_axis)

        slice_object = self._require_object(slice_object_id)
        params = dict(slice_object.parameters or {})
        points = self._normalize_polyline_points(params.get("points", []))
        draw_plane = self._normalize_draw_plane(params.get("draw_plane", "xoy"))
        line_axes, section_axis = self._draw_plane_axes(draw_plane)
        expected_axis = np.zeros(3, dtype=float)
        expected_axis[section_axis] = 1.0
        if abs(float(np.dot(axis_vector, expected_axis))) < 0.999:
            axis_name = {0: "X", 1: "Y", 2: "Z"}[section_axis]
            raise ValueError(f"当前折线剖面仅支持绕 {axis_name} 轴旋转。")
        center_uv = np.mean(points[:, list(line_axes)], axis=0)
        angle_rad = np.deg2rad(float(angle_deg))
        cos_value = float(np.cos(angle_rad))
        sin_value = float(np.sin(angle_rad))
        rotation = np.array([[cos_value, -sin_value], [sin_value, cos_value]], dtype=float)
        shifted_uv = points[:, list(line_axes)] - center_uv[None, :]
        rotated_uv = shifted_uv @ rotation.T + center_uv[None, :]
        rotated_points = points.copy()
        rotated_points[:, line_axes[0]] = rotated_uv[:, 0]
        rotated_points[:, line_axes[1]] = rotated_uv[:, 1]

        section_max = float(params.get("section_max", params.get("top_z")))
        section_min = float(params.get("section_min", params.get("bottom_z")))
        source_object = self._resolve_slice_source_object(slice_object)
        return self.create_polyline_section(
            source_object.object_id,
            rotated_points,
            top_z=section_max,
            bottom_z=section_min,
            draw_plane=draw_plane,
            line_step=float(params["line_step"]),
            vertical_samples=int(params["vertical_samples"]),
            render=render,
            add_to_scene=add_to_scene,
            object_id=object_id,
        )

    def _rotate_vector(self, vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        v = np.asarray(vector, dtype=float).reshape(3)
        k = np.asarray(axis, dtype=float).reshape(3)
        k_norm = float(np.linalg.norm(k))
        if k_norm <= 1e-12:
            raise ValueError("旋转轴无效。")
        k = k / k_norm
        cos_value = float(np.cos(angle_rad))
        sin_value = float(np.sin(angle_rad))
        return (
            v * cos_value
            + np.cross(k, v) * sin_value
            + k * float(np.dot(k, v)) * (1.0 - cos_value)
        )

    def _normalize_polyline_points(self, polyline_points) -> np.ndarray:
        points = np.asarray(polyline_points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            raise ValueError("折线至少需要两个点。")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0], dtype=float)])
        if points.shape[1] != 3:
            raise ValueError("折线点必须是二维或三维坐标。")
        return points

    def _principal_direction_xy(self, points: np.ndarray) -> np.ndarray:
        xy = np.asarray(points[:, :2], dtype=float)
        centered = xy - xy.mean(axis=0, keepdims=True)
        norm_centered = float(np.linalg.norm(centered))
        if norm_centered <= 1e-12:
            raise ValueError("折线点重合，无法生成平面切片。")

        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values.size > 0 and float(singular_values[0]) > 1e-12:
            direction = np.asarray(vh[0], dtype=float)
        else:
            direction = np.asarray(xy[-1] - xy[0], dtype=float)

        norm_direction = float(np.linalg.norm(direction))
        if norm_direction <= 1e-12:
            raise ValueError("折线方向无效，无法生成平面切片。")
        return direction / norm_direction

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

    def _clip_regular_grid_index_range(
        self,
        data: pv.ImageData,
        index_bounds: tuple[int, int, int, int, int, int],
    ) -> pv.ImageData:
        values = np.asarray(index_bounds, dtype=int).reshape(-1)
        if values.size != 6:
            raise ValueError("格点索引范围必须包含 6 个整数。")
        dimensions = np.asarray(data.dimensions, dtype=int)
        voi: list[int] = []
        for axis in range(3):
            axis_min = int(min(values[axis * 2], values[axis * 2 + 1]))
            axis_max = int(max(values[axis * 2], values[axis * 2 + 1]))
            axis_min = max(0, min(axis_min, int(dimensions[axis]) - 1))
            axis_max = max(0, min(axis_max, int(dimensions[axis]) - 1))
            if axis_min > axis_max:
                raise ValueError("该操作结果为空。")
            voi.extend([axis_min, axis_max])
        return data.extract_subset(tuple(voi), rebase_coordinates=False)

    def _mask_regular_grid_with_polygon(
        self,
        data: pv.ImageData,
        polygon_points: np.ndarray,
        *,
        draw_plane: str,
    ) -> pv.ImageData:
        points = np.asarray(polygon_points, dtype=float)
        plane = self._normalize_draw_plane(draw_plane)
        line_axes, section_axis = self._draw_plane_axes(plane)
        polygon_2d = points[:, list(line_axes)]
        if polygon_2d.shape[0] < 3:
            raise ValueError("掩膜边界至少需要三个点。")
        if not np.allclose(polygon_2d[0], polygon_2d[-1]):
            polygon_2d = np.vstack([polygon_2d, polygon_2d[0]])

        dims = np.asarray(data.dimensions, dtype=int)
        origin = np.asarray(data.origin, dtype=float)
        spacing = np.asarray(data.spacing, dtype=float)
        axis_u, axis_v = line_axes
        u_axis = origin[axis_u] + spacing[axis_u] * np.arange(dims[axis_u], dtype=float)
        v_axis = origin[axis_v] + spacing[axis_v] * np.arange(dims[axis_v], dtype=float)
        gu, gv = np.meshgrid(u_axis, v_axis, indexing="ij")
        inside_2d = self._points_inside_polygon_xy(
            gu.ravel(order="F"),
            gv.ravel(order="F"),
            polygon_2d,
        ).reshape((dims[axis_u], dims[axis_v]), order="F")

        if not np.any(inside_2d):
            raise ValueError("掩膜区域为空，请检查边界是否位于数据范围内。")

        if section_axis == 2:
            mask_points_3d = np.repeat(inside_2d[:, :, None], dims[2], axis=2)
        elif section_axis == 1:
            mask_points_3d = np.repeat(inside_2d[:, None, :], dims[1], axis=1)
        else:
            mask_points_3d = np.repeat(inside_2d[None, :, :], dims[0], axis=0)

        mask_points = mask_points_3d.ravel(order="F")
        result = data.copy(deep=True)
        for name in list(result.point_data.keys()):
            values = np.asarray(result.point_data[name])
            flat = values.reshape(-1).astype(float, copy=True)
            if flat.size != mask_points.size:
                continue
            flat[~mask_points] = np.nan
            result.point_data[name] = flat

        cell_dims = np.maximum(dims - 1, 0)
        if np.all(cell_dims > 0):
            mask_cells_2d = (
                inside_2d[:-1, :-1]
                & inside_2d[1:, :-1]
                & inside_2d[:-1, 1:]
                & inside_2d[1:, 1:]
            )
            if section_axis == 2:
                mask_cells_3d = np.repeat(mask_cells_2d[:, :, None], cell_dims[2], axis=2)
            elif section_axis == 1:
                mask_cells_3d = np.repeat(mask_cells_2d[:, None, :], cell_dims[1], axis=1)
            else:
                mask_cells_3d = np.repeat(mask_cells_2d[None, :, :], cell_dims[0], axis=0)
            mask_cells = mask_cells_3d.ravel(order="F")
            for name in list(result.cell_data.keys()):
                values = np.asarray(result.cell_data[name])
                flat = values.reshape(-1).astype(float, copy=True)
                if flat.size != mask_cells.size:
                    continue
                flat[~mask_cells] = np.nan
                result.cell_data[name] = flat
        return result

    def _apply_drillhole_mask(self, data: pv.DataSet, tube_surface: pv.PolyData) -> pv.DataSet:
        if isinstance(data, pv.ImageData):
            return self._mask_regular_grid_with_tube(data, tube_surface)
        return self._clip_mesh_with_tube(data, tube_surface)

    def _mask_regular_grid_with_tube(self, data: pv.ImageData, tube_surface: pv.PolyData) -> pv.ImageData:
        result = data.copy(deep=True)
        point_distance_data = data.compute_implicit_distance(tube_surface, inplace=False)
        point_distances = np.asarray(point_distance_data.point_data["implicit_distance"], dtype=float).reshape(-1)
        inside_points = point_distances <= 0.0
        if not np.any(inside_points):
            return result

        for name in list(result.point_data.keys()):
            values = np.asarray(result.point_data[name])
            flat = values.reshape(-1).astype(float, copy=True)
            if flat.size != inside_points.size:
                continue
            flat[inside_points] = np.nan
            result.point_data[name] = flat

        if result.n_cells > 0 and len(result.cell_data.keys()) > 0:
            cell_distance_data = data.cell_centers().compute_implicit_distance(tube_surface, inplace=False)
            cell_distances = np.asarray(cell_distance_data.point_data["implicit_distance"], dtype=float).reshape(-1)
            inside_cells = cell_distances <= 0.0
            for name in list(result.cell_data.keys()):
                values = np.asarray(result.cell_data[name])
                flat = values.reshape(-1).astype(float, copy=True)
                if flat.size != inside_cells.size:
                    continue
                flat[inside_cells] = np.nan
                result.cell_data[name] = flat
        return result

    def _clip_mesh_with_tube(self, data: pv.DataSet, tube_surface: pv.PolyData) -> pv.DataSet:
        if int(getattr(data, "n_cells", 0)) <= 0:
            return data.copy(deep=True)

        working = data.copy(deep=True)
        centers = working.cell_centers()
        center_with_distance = centers.compute_implicit_distance(tube_surface, inplace=False)
        cell_distances = np.asarray(center_with_distance.point_data["implicit_distance"], dtype=float).reshape(-1)
        if cell_distances.size != int(getattr(working, "n_cells", 0)):
            return working

        working.cell_data["__drill_dist__"] = cell_distances
        clipped = working.threshold(
            value=0.0,
            scalars="__drill_dist__",
            preference="cell",
            invert=False,
        )
        if "__drill_dist__" in clipped.cell_data:
            del clipped.cell_data["__drill_dist__"]
        if "__drill_dist__" in clipped.point_data:
            del clipped.point_data["__drill_dist__"]
        return clipped

    def _build_drillhole_tube_display_data(
        self,
        tube_surface: pv.PolyData,
        source_object,
    ) -> tuple[pv.PolyData, str | None, tuple[float, float] | None]:
        scalar_name = source_object.active_scalar
        if scalar_name is None:
            return tube_surface, None, None

        sampled_tube = self._sample_surface_scalar(tube_surface, source_object.data, scalar_name)
        if sampled_tube is None:
            return tube_surface, None, None

        scalar_range = source_object.style.clim or self._extract_data_scalar_range(sampled_tube, scalar_name)
        return sampled_tube, scalar_name, scalar_range

    def _sample_surface_scalar(
        self,
        surface: pv.PolyData,
        source_data: pv.DataSet,
        scalar_name: str,
    ) -> pv.PolyData | None:
        candidates = [source_data]
        if scalar_name in source_data.cell_data and scalar_name not in source_data.point_data:
            try:
                candidates.append(source_data.cell_data_to_point_data(pass_cell_data=True))
            except Exception:
                pass

        for candidate in candidates:
            try:
                sampled = surface.sample(candidate)
            except Exception:
                continue
            if self._has_finite_scalar(sampled, scalar_name):
                return sampled
        return None

    def _extract_data_scalar_range(
        self,
        data: pv.DataSet,
        scalar_name: str,
    ) -> tuple[float, float] | None:
        values = self._extract_scalar_values(data, scalar_name)
        if values is None:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return (float(np.min(finite)), float(np.max(finite)))

    def _has_finite_scalar(self, data: pv.DataSet, scalar_name: str) -> bool:
        values = self._extract_scalar_values(data, scalar_name)
        return values is not None and bool(np.isfinite(values).any())

    def _extract_scalar_values(self, data: pv.DataSet, scalar_name: str) -> np.ndarray | None:
        if scalar_name in data.point_data:
            return np.asarray(data.point_data[scalar_name], dtype=float).reshape(-1)
        if scalar_name in data.cell_data:
            return np.asarray(data.cell_data[scalar_name], dtype=float).reshape(-1)
        return None

    def _add_helper_object(
        self,
        data: pv.DataSet,
        *,
        name: str,
        color: tuple[float, float, float] | None,
        render_mode: str,
        source_object_id: str,
        parameters: dict[str, Any],
        render: bool,
        add_to_scene: bool,
        scalar_name: str | None = None,
        clim: tuple[float, float] | None = None,
        colormap: str = "viridis",
        show_scalar_bar: bool = False,
    ) -> DatasetSceneObject:
        dataset = create_dataset_from_pyvista(
            data,
            source_path="",
            name=name,
            metadata=DatasetMetadata(
                source_path="",
                source_name=name,
                dataset_type="helper",
                source_schema={"operation": "drillhole_helper"},
            ),
            import_spec=None,
        )
        active_scalar = scalar_name if scalar_name in dataset.scalar_fields else None
        scalar_range = clim
        if active_scalar is not None and scalar_range is None:
            scalar_range = dataset.get_scalar_range(active_scalar)
        style = RenderStyle(
            visible=True,
            opacity=1.0,
            color=tuple(float(v) for v in color) if color is not None else None,
            scalar_name=active_scalar,
            colormap=colormap,
            clim=scalar_range,
            show_scalar_bar=bool(show_scalar_bar and active_scalar),
            render_mode=render_mode,
        ).normalized()
        if add_to_scene:
            return self.add_dataset(
                dataset,
                render=render,
                name=name,
                object_type="helper",
                style=style,
                source_object_id=source_object_id,
                parameters=parameters,
            )
        return self.create_dataset_object(
            dataset,
            name=name,
            object_type="helper",
            style=style,
            source_object_id=source_object_id,
            parameters=parameters,
        )

    def _resolve_well_index(
        self,
        well_object,
        *,
        well_index: int | None,
        well_name: str | None,
    ) -> int | None:
        if well_index is not None:
            return int(well_index)
        if well_name is None:
            return None
        schema = dict(getattr(well_object.dataset, "source_schema", {}) or {})
        well_ids = [str(value) for value in schema.get("well_ids", [])]
        if not well_ids:
            raise ValueError("井数据中不存在井号映射。")
        target = str(well_name).strip()
        if target not in well_ids:
            raise ValueError(f"未找到井号：{target}")
        return int(well_ids.index(target))

    def _extract_point_data_array(self, data: pv.DataSet, name: str) -> np.ndarray | None:
        if name not in data.point_data:
            return None
        values = np.asarray(data.point_data[name]).reshape(-1)
        if values.size != int(getattr(data, "n_points", 0)):
            return None
        return values

    def _extract_well_polyline_ids(
        self,
        data: pv.DataSet,
        *,
        selected_indices: np.ndarray,
        well_indices: np.ndarray | None,
        selected_well_index: int | None,
    ) -> np.ndarray | None:
        lines = np.asarray(getattr(data, "lines", np.empty(0)), dtype=np.int64).reshape(-1)
        if lines.size <= 0:
            return None

        selected_set = set(int(value) for value in selected_indices.tolist())
        candidates: list[np.ndarray] = []
        cursor = 0
        while cursor < lines.size:
            count = int(lines[cursor])
            cursor += 1
            if count <= 1 or cursor + count > lines.size:
                break
            ids = np.asarray(lines[cursor : cursor + count], dtype=np.int64)
            cursor += count
            if selected_set:
                overlap = [idx for idx in ids.tolist() if int(idx) in selected_set]
                if len(overlap) >= 2:
                    candidates.append(np.asarray(overlap, dtype=np.int64))
                    continue
            candidates.append(ids)

        if not candidates:
            return None

        if selected_well_index is not None and well_indices is not None:
            scored = []
            for ids in candidates:
                local = well_indices[ids]
                score = int(np.sum(local == selected_well_index))
                scored.append((score, ids))
            scored.sort(key=lambda item: (item[0], item[1].size), reverse=True)
            best = scored[0][1]
            if scored[0][0] >= 2:
                return best
        candidates.sort(key=lambda item: item.size, reverse=True)
        return candidates[0]

    def _sort_well_points(self, points: np.ndarray, depth_values: np.ndarray | None) -> np.ndarray:
        if depth_values is not None:
            finite = np.isfinite(depth_values)
            if np.any(finite):
                fill = np.nanmax(depth_values[finite]) + 1.0
                sortable = np.where(finite, depth_values, fill)
                order = np.argsort(sortable, kind="stable")
                return points[order]
        order = np.argsort(-points[:, 2], kind="stable")
        return points[order]

    def _build_polyline_from_points(self, points: np.ndarray) -> pv.PolyData:
        line = pv.PolyData(np.asarray(points, dtype=float))
        count = int(line.n_points)
        if count < 2:
            raise ValueError("井轨迹点不足两个。")
        line.lines = np.concatenate(([count], np.arange(count, dtype=np.int64)))
        return line

    def _points_inside_polygon_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        polygon_xy: np.ndarray,
    ) -> np.ndarray:
        inside = np.zeros(x.shape[0], dtype=bool)
        px = polygon_xy[:, 0]
        py = polygon_xy[:, 1]
        epsilon = 1e-12
        for idx in range(len(px) - 1):
            x0, y0 = float(px[idx]), float(py[idx])
            x1, y1 = float(px[idx + 1]), float(py[idx + 1])
            intersects = ((y0 > y) != (y1 > y)) & (
                x < (x1 - x0) * (y - y0) / ((y1 - y0) + epsilon) + x0
            )
            inside ^= intersects
        return inside

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
