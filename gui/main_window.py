"""
三维正反演可视化软件主窗口。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt6.QtCore import Qt, QThreadPool, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QDockWidget, QMainWindow, QMessageBox

from gui.SceneManagerPanel import SceneManagerPanel
from gui.axis_scale_component import AxisScaleComponent
from gui.clip_panel import ClipPanel
from gui.interactive_view import InteractiveView
from gui.professional_toolbar import ProfessionalToolbar
from gui.property_panel import PropertyPanel
from gui.slice_panel import SlicePanel
from gui.task_runner import Worker
from gui.view_axes_2d import ViewAxes2D
from services import ImportService, ProjectService, SceneService


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维正反演可视化软件")
        self.setGeometry(80, 80, 1680, 960)

        self.thread_pool = QThreadPool.globalInstance()
        self._workers: set[Worker] = set()
        self._selection_outline_actors = []
        self.selected_object_id: str | None = None
        self.initial_workspace_bounds = np.array([-100.0, 100.0, -100.0, 100.0, -50.0, 50.0], dtype=float)

        self.import_service = ImportService()
        self.project_service = ProjectService()
        self.scene_service = SceneService()

        self._create_menu_bar()
        self._create_status_bar()
        self._create_main_widget()

        self.project = self.scene_service.project
        self.statusBar().showMessage("就绪")

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        actions = [
            ("导入数据", self.import_data),
            ("打开工程", self.open_project),
            ("保存工程", self.save_project),
            ("导出截图", self.export_screenshot),
            ("导出当前对象", self.export_selected_object),
        ]
        for text, callback in actions:
            action = QAction(text, self)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("视图")
        view_actions = [
            ("重置视图", self.reset_view),
            ("切换方向组件", self.toggle_axes),
            ("切换网格", self.toggle_grid),
            ("切换原点坐标轴", self.toggle_origin_axes),
            ("切换坐标刻度", self.toggle_axis_scales),
            ("清除选中高亮", self.clear_selection_outline),
        ]
        for text, callback in view_actions:
            action = QAction(text, self)
            action.triggered.connect(callback)
            view_menu.addAction(action)

        tools_menu = menubar.addMenu("工具")
        clear_derived_action = QAction("清除派生对象", self)
        clear_derived_action.triggered.connect(self.clear_derived_objects)
        tools_menu.addAction(clear_derived_action)
        open_slice_action = QAction("打开切片窗口", self)
        open_slice_action.triggered.connect(self.show_slice_window)
        tools_menu.addAction(open_slice_action)
        open_clip_action = QAction("打开裁剪窗口", self)
        open_clip_action.triggered.connect(self.show_clip_window)
        tools_menu.addAction(open_clip_action)

        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        self.statusBar().showMessage("就绪")

    def _create_main_widget(self):
        pv.set_plot_theme("default")
        self.plotter = InteractiveView(
            self,
            workspace_bounds=self.initial_workspace_bounds.copy(),
            background_color="white",
        )
        self.setCentralWidget(self.plotter)
        self.scene_service.set_plotter(self.plotter)
        if hasattr(self.plotter, "status_message"):
            self.plotter.status_message.connect(self.statusBar().showMessage)

        self.toolbar = ProfessionalToolbar(self)
        self.toolbar.action_triggered.connect(self._on_toolbar_action)
        self.addToolBar(self.toolbar)

        self.scene_manager = SceneManagerPanel(self)
        self.scene_manager.set_plotter(self.plotter)
        self.scene_manager.visibilityChanged.connect(self._on_visibility_changed)
        self.scene_manager.opacityChanged.connect(self._on_opacity_changed)
        self.scene_manager.objectSelected.connect(self.on_object_selected)
        self.scene_manager.renameRequested.connect(self._on_rename_requested)
        self.scene_manager.deleteRequested.connect(self._on_delete_requested)
        self.scene_dock = self._add_dock("场景管理", self.scene_manager, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.property_panel = PropertyPanel(self)
        self.property_panel.scalarChanged.connect(self._on_scalar_changed)
        self.property_panel.renderModeChanged.connect(
            lambda object_id, mode: self._apply_style_update(object_id, render_mode=mode)
        )
        self.property_panel.colormapChanged.connect(
            lambda object_id, cmap: self._apply_style_update(object_id, colormap=cmap)
        )
        self.property_panel.opacityChanged.connect(self._on_opacity_changed)
        self.property_panel.climChanged.connect(
            lambda object_id, clim: self._apply_style_update(object_id, clim=clim)
        )
        self.property_panel.scalarBarChanged.connect(
            lambda object_id, value: self._apply_style_update(object_id, show_scalar_bar=value)
        )
        self.property_panel.thresholdChanged.connect(
            lambda object_id, value: self._apply_style_update(object_id, threshold_range=value)
        )
        self.property_panel.isosurfaceRequested.connect(self._create_isosurface)
        self.property_panel.idwRequested.connect(self._create_idw_grid)
        self.property_dock = self._add_dock("属性控制", self.property_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.slice_panel = SlicePanel()
        self.slice_panel.axisBatchSliceRequested.connect(self._create_axis_slice_batch)
        self.slice_panel.orthogonalSliceRequested.connect(self._create_orthogonal_slice)
        self.slice_panel.planeSliceRequested.connect(self._create_plane_slice)
        self.slice_panel.clearDerivedRequested.connect(self.clear_derived_objects)
        self.slice_dock = self._add_dock("切片窗口", self.slice_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.slice_window = self.slice_dock

        self.clip_panel = ClipPanel()
        self.clip_panel.clipRequested.connect(self._create_clip)
        self.clip_panel.clearDerivedRequested.connect(self.clear_derived_objects)
        self.clip_dock = self._add_dock("裁剪窗口", self.clip_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.clip_window = self.clip_dock

        self.splitDockWidget(self.property_dock, self.slice_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self.slice_dock, self.clip_dock, Qt.Orientation.Vertical)

        self.view_axes = ViewAxes2D(self.plotter, size=96)
        self.view_axes.setParent(self.plotter)
        self.view_axes.raise_()
        self.axis_scale_component = AxisScaleComponent(self.plotter)

        def update_view_axes():
            try:
                camera = self.plotter.renderer.GetActiveCamera()
                position = np.array(camera.GetPosition())
                focal_point = np.array(camera.GetFocalPoint())
                view_up = np.array(camera.GetViewUp())
                direction = position - focal_point
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    self.view_axes.update_camera_direction(direction / norm, view_up)
            except Exception:
                return

        self.plotter.view_changed.connect(update_view_axes)
        QTimer.singleShot(100, lambda: [update_view_axes(), self._update_view_axes_position()])

    def _add_dock(self, title: str, widget, area: Qt.DockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _on_toolbar_action(self, action_name: str):
        action_map = {
            "import_data": self.import_data,
            "open_project": self.open_project,
            "save_project": self.save_project,
            "reset_view": self.reset_view,
            "clear_derived": self.clear_derived_objects,
            "open_slice_window": self.show_slice_window,
            "open_clip_window": self.show_clip_window,
            "export_screenshot": self.export_screenshot,
            "export_object": self.export_selected_object,
        }
        callback = action_map.get(action_name)
        if callback:
            callback()

    def _run_worker(self, description: str, func, on_success):
        worker = Worker(func)
        self._workers.add(worker)
        self.statusBar().showMessage(description)

        def finish(result):
            self._workers.discard(worker)
            on_success(result)

        def fail(message: str):
            self._workers.discard(worker)
            QMessageBox.critical(self, "操作失败", message)
            self.statusBar().showMessage(message, 5000)

        worker.signals.finished.connect(finish)
        worker.signals.failed.connect(fail)
        self.thread_pool.start(worker)

    def _show_dock(self, dock: QDockWidget | None):
        if dock is None:
            return
        dock.show()
        dock.raise_()
        dock.activateWindow()
        if dock.isFloating():
            dock.raise_()

    def show_slice_window(self):
        self._show_dock(getattr(self, "slice_dock", None))
        if hasattr(self, "slice_panel"):
            self.slice_panel.focus_slice_controls()
        self.statusBar().showMessage("已打开切片窗口", 2500)

    def show_clip_window(self):
        self._show_dock(getattr(self, "clip_dock", None))
        if hasattr(self, "clip_panel"):
            self.clip_panel.focus_clip_controls()
        self.statusBar().showMessage("已打开裁剪窗口", 2500)

    def import_data(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入三维数据",
            "",
            (
                "支持的文件 (*.vtk *.vtu *.vtr *.vts *.vtm *.csv *.xyz *.dat);;"
                "VTK 文件 (*.vtk *.vtu *.vtr *.vts *.vtm);;"
                "文本文件 (*.csv *.xyz *.dat);;所有文件 (*)"
            ),
        )
        if not file_paths:
            return

        self._run_worker(
            "正在导入数据...",
            lambda: self.import_service.import_files(file_paths),
            self._on_import_finished,
        )

    def _on_import_finished(self, result):
        imported, failures = result
        added_objects = []
        for dataset in imported:
            scene_object = self.scene_service.add_dataset(dataset, render=True)
            self.scene_manager.add_object(scene_object)
            added_objects.append(scene_object)

        if added_objects:
            self._fit_scene_to_objects()
            self.plotter.reset_camera()
            self.plotter.render()

        if failures:
            message = "\n".join(f"{Path(path).name}: {exc}" for path, exc in failures)
            QMessageBox.warning(self, "导入完成，但存在错误", message)

        self.statusBar().showMessage(
            f"成功导入 {len(added_objects)} 个数据集。",
            4000,
        )

    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开工程",
            "",
            "可视化工程 (*.json);;所有文件 (*)",
        )
        if not file_path:
            return
        try:
            payload = self.project_service.load_project(file_path)
            self.scene_service.load_from_payload(payload, self.import_service)
            self.scene_manager.rebuild(self.scene_service.all_objects())
            self._fit_scene_to_objects()
            if payload.get("camera_state"):
                self.plotter.set_camera_info(payload["camera_state"])
            else:
                self.plotter.reset_camera()
            self.project = self.scene_service.project
            self.statusBar().showMessage(f"已打开工程：{Path(file_path).name}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "打开工程失败", str(exc))

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存工程",
            "",
            "可视化工程 (*.json);;所有文件 (*)",
        )
        if not file_path:
            return
        try:
            camera_state = self.plotter.get_camera_info()
            self.project_service.save_project(
                file_path,
                project_name=self.scene_service.project.name,
                metadata=self.scene_service.project.metadata,
                camera_state=camera_state,
                objects=self.scene_service.serialize_scene(),
            )
            self.statusBar().showMessage(f"工程已保存：{file_path}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "保存工程失败", str(exc))

    def export_screenshot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出截图",
            "",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;BMP 图片 (*.bmp)",
        )
        if not file_path:
            return
        try:
            self.plotter.screenshot(file_path)
            self.statusBar().showMessage(f"截图已导出：{file_path}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "导出截图失败", str(exc))

    def export_selected_object(self):
        scene_object = self._get_selected_scene_object()
        if scene_object is None:
            QMessageBox.information(self, "导出", "请先选择要导出的对象。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前对象",
            f"{scene_object.name}.vtk",
            "VTK 文件 (*.vtk *.vtp *.vtu);;CSV 文件 (*.csv)",
        )
        if not file_path:
            return
        try:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".csv":
                self._export_dataset_csv(scene_object, file_path)
            else:
                scene_object.data.save(file_path)
            self.statusBar().showMessage(f"对象已导出：{file_path}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _export_dataset_csv(self, scene_object, file_path: str):
        points = np.asarray(scene_object.data.points)
        headers = ["x", "y", "z"]
        columns = [points[:, 0], points[:, 1], points[:, 2]]
        for name in scene_object.data.point_data.keys():
            headers.append(name)
            columns.append(np.asarray(scene_object.data.point_data[name]).reshape(-1))
        matrix = np.column_stack(columns)
        header = ",".join(headers)
        np.savetxt(file_path, matrix, delimiter=",", header=header, comments="")

    def reset_view(self):
        self._fit_scene_to_objects()
        self.plotter.reset_camera()
        self.statusBar().showMessage("视图已重置", 2500)

    def toggle_axes(self):
        self.view_axes.setVisible(not self.view_axes.isVisible())
        self.statusBar().showMessage(
            "方向组件已显示" if self.view_axes.isVisible() else "方向组件已隐藏",
            2000,
        )

    def toggle_origin_axes(self):
        self.plotter.toggle_origin_axes()
        self.statusBar().showMessage(
            "原点坐标轴已显示" if self.plotter.get_show_origin_axes() else "原点坐标轴已隐藏",
            2000,
        )

    def toggle_axis_scales(self):
        visible = self.axis_scale_component.toggle_visible()
        self.statusBar().showMessage(
            "坐标刻度已显示" if visible else "坐标刻度已隐藏",
            2000,
        )

    def toggle_grid(self):
        self.plotter.toggle_grid()
        self.statusBar().showMessage(
            "网格已显示" if self.plotter.get_show_grid() else "网格已隐藏",
            2000,
        )

    def clear_selection_outline(self):
        for actor in list(self._selection_outline_actors):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._selection_outline_actors = []
        self.plotter.render()

    def on_object_selected(self, object_id: str):
        self.selected_object_id = object_id
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None:
            return
        source_object = (
            self.scene_service.get_object(scene_object.source_object_id)
            if scene_object.source_object_id
            else scene_object
        )
        self.property_panel.set_scene_object(scene_object)
        self.slice_panel.set_scene_object(source_object)
        self.clip_panel.set_scene_object(source_object)
        highlight_targets = [scene_object]
        if source_object is not None and source_object.object_id != scene_object.object_id:
            highlight_targets.append(source_object)
        self._highlight_objects(highlight_targets)
        self.statusBar().showMessage(f"已选中：{scene_object.name}", 2500)

    def _highlight_objects(self, scene_objects):
        self.clear_selection_outline()
        colors = ["yellow", "orange"]
        for index, scene_object in enumerate(scene_objects):
            self._highlight_single_object(scene_object, colors[min(index, len(colors) - 1)])

    def _highlight_single_object(self, scene_object, color: str):
        try:
            data = scene_object.data
            if isinstance(data, pv.ImageData):
                mesh = data.extract_surface()
            elif hasattr(data, "extract_surface") and not isinstance(data, pv.PolyData):
                mesh = data.extract_surface()
            else:
                mesh = data
            edges = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=False,
            )
            if edges.n_points == 0:
                outline = mesh.outline()
            else:
                outline = edges
            actor = self.plotter.add_mesh(
                outline,
                color=color,
                line_width=3,
                name=f"selection_outline_{scene_object.object_id}",
                pickable=False,
            )
            self._selection_outline_actors.append(actor)
            self.plotter.render()
        except Exception:
            return

    def _fit_scene_to_objects(self):
        bounds_array = []
        for scene_object in self.scene_service.all_objects():
            if scene_object.dataset is not None:
                bounds_array.append(scene_object.bounds)

        if not bounds_array:
            self.plotter.set_workspace_bounds(self.initial_workspace_bounds.copy())
            return

        bounds = np.array(bounds_array, dtype=float)
        mins = np.min(bounds[:, [0, 2, 4]], axis=0)
        maxs = np.max(bounds[:, [1, 3, 5]], axis=0)
        ranges = np.maximum(maxs - mins, 1.0)
        padding = np.maximum(ranges * 0.05, 1.0)
        workspace_bounds = np.array(
            [
                mins[0] - padding[0],
                maxs[0] + padding[0],
                mins[1] - padding[1],
                maxs[1] + padding[1],
                mins[2] - padding[2],
                maxs[2] + padding[2],
            ],
            dtype=float,
        )
        self.plotter.set_workspace_bounds(workspace_bounds)

    def _get_selected_scene_object(self):
        if not self.selected_object_id:
            return None
        return self.scene_service.get_object(self.selected_object_id)

    def _apply_style_update(self, object_id: str, **updates):
        scene_object = self.scene_service.update_style(object_id, **updates)
        if scene_object is None:
            return
        self.scene_manager.refresh_object(scene_object)
        if self.selected_object_id == object_id:
            self.property_panel.set_scene_object(scene_object)

    def _on_scalar_changed(self, object_id: str, scalar_name: str):
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None:
            return
        scalar = scalar_name or None
        clim = scene_object.dataset.get_scalar_range(scalar) if scalar is not None else None
        self._apply_style_update(object_id, scalar_name=scalar, clim=clim)

    def _on_visibility_changed(self, object_id: str, visible: bool):
        scene_object = self.scene_service.set_visibility(object_id, visible)
        if scene_object is not None:
            self.scene_manager.refresh_object(scene_object)

    def _on_opacity_changed(self, object_id: str, opacity: float):
        scene_object = self.scene_service.set_opacity(object_id, opacity)
        if scene_object is None:
            return
        self.scene_manager.on_scene_opacity_changed(object_id, opacity)
        if self.selected_object_id == object_id:
            self.property_panel.set_scene_object(scene_object)

    def _on_rename_requested(self, object_id: str, new_name: str):
        scene_object = self.scene_service.rename_object(object_id, new_name)
        self.scene_manager.refresh_object(scene_object)
        if self.selected_object_id == object_id:
            self.property_panel.set_scene_object(scene_object)

    def _on_delete_requested(self, object_id: str):
        self.scene_service.remove_object(object_id)
        self.scene_manager.remove_object(object_id)
        if self.selected_object_id == object_id:
            self.selected_object_id = None
            self.property_panel.set_scene_object(None)
            self.slice_panel.set_scene_object(None)
            self.clip_panel.set_scene_object(None)
            self.clear_selection_outline()

    def clear_derived_objects(self, source_object_id: str | None = None):
        if source_object_id == "":
            source_object_id = None
        self.scene_service.clear_derived_objects(source_object_id=source_object_id)
        self.scene_manager.rebuild(self.scene_service.all_objects())
        if self.selected_object_id and self.scene_service.get_object(self.selected_object_id) is None:
            self.selected_object_id = None
            self.property_panel.set_scene_object(None)
            self.slice_panel.set_scene_object(None)
            self.clip_panel.set_scene_object(None)
        self.clear_selection_outline()
        self.statusBar().showMessage("派生对象已清除", 2500)

    def _create_axis_slice(self, object_id: str, axis: str, position: float):
        self._run_worker(
            f"正在生成 {axis.upper()} 向切片...",
            lambda: self.scene_service.create_axis_slice(
                object_id,
                axis,
                position,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_axis_slice_batch(self, object_id: str, params: dict):
        self._run_worker(
            "正在按步长批量生成切片...",
            lambda: self.scene_service.create_axis_slice_batch(
                object_id,
                params["axis"],
                params["start"],
                params["end"],
                params["step"],
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_orthogonal_slice(self, object_id: str, params: dict):
        self._run_worker(
            "正在生成三向切片...",
            lambda: self.scene_service.create_orthogonal_slice(
                object_id,
                params["x"],
                params["y"],
                params["z"],
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_plane_slice(self, object_id: str, params: dict):
        self._run_worker(
            "正在生成任意平面切片...",
            lambda: self.scene_service.create_plane_slice(
                object_id,
                params["origin"],
                params["normal"],
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_clip(self, object_id: str, bounds):
        self._run_worker(
            "正在应用裁剪...",
            lambda: self.scene_service.create_clip_box(
                object_id,
                bounds,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_isosurface(self, object_id: str, value: float):
        self._run_worker(
            "正在生成等值面...",
            lambda: self.scene_service.create_isosurface(
                object_id,
                value,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _create_idw_grid(self, object_id: str, params: dict):
        self._run_worker(
            "正在进行点集插值...",
            lambda: self.scene_service.interpolate_point_dataset_to_grid(
                object_id,
                dimensions=params["dimensions"],
                power=params["power"],
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _on_created_objects_ready(self, result):
        if result is None:
            return
        scene_objects = result if isinstance(result, list) else [result]
        for scene_object in scene_objects:
            self.scene_service.add_object(scene_object, render=True)
            self.scene_manager.add_object(scene_object)
        if len(scene_objects) == 1:
            self.statusBar().showMessage(f"已生成：{scene_objects[0].name}", 3500)
        else:
            self.statusBar().showMessage(f"已批量生成 {len(scene_objects)} 个切片对象", 4000)

    def _update_view_axes_position(self):
        if hasattr(self, "view_axes") and hasattr(self, "plotter"):
            plotter_size = self.plotter.size()
            axes_size = self.view_axes.size().width()
            margin = 10
            self.view_axes.move(plotter_size.width() - axes_size - margin, margin)
            self.view_axes.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_view_axes_position()

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "三维正反演可视化软件\n\n"
            "版本：1.0\n\n"
            "主要功能：\n"
            "- 规则体数据体渲染\n"
            "- 非规则网格与曲面可视化\n"
            "- 轴向切片、三向切片、任意平面切片\n"
            "- 框裁剪与等值面生成\n"
            "- CSV/XYZ/DAT 与 VTK 系列数据导入\n"
            "- 工程保存与恢复",
        )
