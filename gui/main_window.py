"""
三维正反演可视化软件主窗口。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt6.QtCore import Qt, QThreadPool, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QDialog, QDockWidget, QInputDialog, QMainWindow, QMessageBox

from gui.SceneManagerPanel import SceneManagerPanel
from gui.axis_scale_component import AxisScaleComponent
from gui.clip_panel import ClipPanel
from gui.interactive_view import InteractiveView
from gui.professional_toolbar import ProfessionalToolbar
from gui.property_panel import PropertyPanel
from gui.slice_panel import SlicePanel
from gui.task_runner import Worker
from gui.view_axes_2d import ViewAxes2D
from gui.well_log_import_dialog import WellLogImportDialog
from services import ImportService, ProjectService, SceneService

WELL_LOG_OBJECT_TYPE = "drillhole"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维正反演可视化软件")
        self.setGeometry(80, 80, 1680, 960)

        self.thread_pool = QThreadPool.globalInstance()
        self._workers: set[Worker] = set()
        self._selection_outline_actors = []
        self._show_view_axes = True
        self._show_selection_highlight = True
        self.selected_object_id: str | None = None
        self._grid_pick_active = False
        self._polyline_owner: str | None = None
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
            ("导入测井数据", self.import_well_log_data),
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

        reset_view_action = QAction("重置视图", self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

        self.toggle_axes_action = QAction("方向组件", self)
        self.toggle_axes_action.setCheckable(True)
        self.toggle_axes_action.toggled.connect(self._set_axes_visible)
        view_menu.addAction(self.toggle_axes_action)

        self.toggle_axis_scales_action = QAction("坐标刻度", self)
        self.toggle_axis_scales_action.setCheckable(True)
        self.toggle_axis_scales_action.toggled.connect(self._set_axis_scales_visible)
        view_menu.addAction(self.toggle_axis_scales_action)

        self.toggle_selection_highlight_action = QAction("选中高亮", self)
        self.toggle_selection_highlight_action.setCheckable(True)
        self.toggle_selection_highlight_action.toggled.connect(self._set_selection_highlight_visible)
        view_menu.addAction(self.toggle_selection_highlight_action)

        self.toggle_scene_tree_action = QAction("场景树", self)
        self.toggle_scene_tree_action.setCheckable(True)
        self.toggle_scene_tree_action.toggled.connect(self._set_scene_tree_visible)
        view_menu.addAction(self.toggle_scene_tree_action)
        view_menu.addSeparator()
        self.toggle_scalar_bar_action = QAction("显示色标", self)
        self.toggle_scalar_bar_action.setCheckable(True)
        self.toggle_scalar_bar_action.setEnabled(False)
        self.toggle_scalar_bar_action.toggled.connect(self._toggle_selected_scalar_bar)
        view_menu.addAction(self.toggle_scalar_bar_action)

        tools_menu = menubar.addMenu("工具")
        clear_derived_action = QAction("清除派生对象", self)
        clear_derived_action.triggered.connect(self.clear_derived_objects)
        tools_menu.addAction(clear_derived_action)
        drillhole_action = QAction("执行钻孔映射", self)
        drillhole_action.triggered.connect(self.run_drillhole_mapping)
        tools_menu.addAction(drillhole_action)
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
        if hasattr(self.plotter, "polyline_changed"):
            self.plotter.polyline_changed.connect(self._on_polyline_points_changed)
        if hasattr(self.plotter, "polyline_finished"):
            self.plotter.polyline_finished.connect(self._on_polyline_finished)
        if hasattr(self.plotter, "polyline_cancelled"):
            self.plotter.polyline_cancelled.connect(self._on_polyline_cancelled)

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
        self.scene_manager.openPropertyRequested.connect(self._open_property_from_scene_manager)
        self.scene_manager.sliceMoveRequested.connect(self._request_slice_move)
        self.scene_manager.sliceTiltRequested.connect(self._request_slice_tilt)
        self.scene_dock = self._add_dock("场景管理", self.scene_manager, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.scene_dock.visibilityChanged.connect(self._on_scene_tree_dock_visibility_changed)

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
        self.slice_panel.polylineDrawingRequested.connect(self._start_polyline_drawing)
        self.slice_panel.polylineDrawingCancelled.connect(self._cancel_polyline_drawing)
        self.slice_panel.polylineSectionRequested.connect(self._create_polyline_section)
        self.slice_panel.clearDerivedRequested.connect(self.clear_derived_objects)
        self.slice_dock = self._add_dock("切片窗口", self.slice_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.slice_window = self.slice_dock

        self.clip_panel = ClipPanel()
        self.clip_panel.maskDrawingStartRequested.connect(self._start_mask_drawing)
        self.clip_panel.maskDrawingCancelRequested.connect(self._cancel_polyline_drawing)
        self.clip_panel.maskClipRequested.connect(self._create_mask_clip)
        self.clip_panel.clearDerivedRequested.connect(self.clear_derived_objects)
        self.clip_dock = self._add_dock("裁剪窗口", self.clip_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.clip_window = self.clip_dock

        self.tabifyDockWidget(self.property_dock, self.slice_dock)
        self.tabifyDockWidget(self.property_dock, self.clip_dock)
        self.property_dock.raise_()
        self.slice_dock.hide()
        self.clip_dock.hide()

        self.view_axes = ViewAxes2D(self.plotter, size=96)
        self.view_axes.setParent(self.plotter)
        self.view_axes.raise_()
        self.axis_scale_component = AxisScaleComponent(self.plotter)
        self._sync_view_menu_actions()

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
            "import_well_log_data": self.import_well_log_data,
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

    def show_property_window(self):
        self._show_dock(getattr(self, "property_dock", None))
        self.statusBar().showMessage("已打开属性控制窗口", 2500)

    def _open_property_from_scene_manager(self, object_id: str):
        self.on_object_selected(object_id)
        self.show_property_window()

    def _request_slice_move(self, slice_object_id: str):
        slice_object = self.scene_service.get_object(slice_object_id)
        if slice_object is None or not self._is_slice_like_object(slice_object):
            return
        component_axis = self._pick_slice_component_axis(slice_object)
        if component_axis == "__cancel__":
            return

        offset, ok = QInputDialog.getDouble(
            self,
            "平移切片",
            "沿切片法向平移距离（正负均可）：",
            0.0,
            -1e9,
            1e9,
            3,
        )
        if not ok:
            return

        self._run_worker(
            "正在平移切片...",
            lambda: self.scene_service.move_slice(
                slice_object_id,
                offset,
                component_axis=component_axis,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _request_slice_tilt(self, slice_object_id: str):
        slice_object = self.scene_service.get_object(slice_object_id)
        if slice_object is None or not self._is_slice_like_object(slice_object):
            return

        component_axis = self._pick_slice_component_axis(slice_object)
        if component_axis == "__cancel__":
            return

        tilt_axes = self._available_tilt_axes(slice_object, component_axis=component_axis)
        if not tilt_axes:
            QMessageBox.information(self, "倾斜切片", "当前切片没有可用的倾斜轴。")
            return
        axis_labels = [axis.upper() for axis in tilt_axes]
        selected_axis, axis_ok = QInputDialog.getItem(
            self,
            "倾斜切片",
            "选择倾斜旋转轴：",
            axis_labels,
            0,
            False,
        )
        if not axis_ok:
            return

        angle_deg, angle_ok = QInputDialog.getDouble(
            self,
            "倾斜切片",
            "输入倾斜角度（度）：",
            15.0,
            -89.0,
            89.0,
            2,
        )
        if not angle_ok:
            return

        self._run_worker(
            "正在倾斜切片...",
            lambda: self.scene_service.tilt_slice(
                slice_object_id,
                angle_deg=angle_deg,
                tilt_axis=selected_axis.lower(),
                component_axis=component_axis,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _pick_slice_component_axis(self, slice_object):
        params = dict(getattr(slice_object, "parameters", {}) or {})
        kind = str(params.get("kind") or "").lower()
        if kind != "orthogonal":
            return None
        axis_label, ok = QInputDialog.getItem(
            self,
            "选择切片分量",
            "三向切片包含 X/Y/Z 三个分量，请先选择要操作的分量：",
            ["X", "Y", "Z"],
            2,
            False,
        )
        if not ok:
            return "__cancel__"
        return axis_label.lower()

    def _available_tilt_axes(self, slice_object, component_axis: str | None = None) -> list[str]:
        kind = str((getattr(slice_object, "parameters", {}) or {}).get("kind") or "").lower()
        if kind == "polyline":
            return ["z"]
        fixed_axis = None
        if kind == "axis":
            fixed_axis = str(slice_object.parameters.get("axis") or "").lower()
        elif kind == "orthogonal":
            fixed_axis = (component_axis or "").lower()

        if fixed_axis in {"x", "y", "z"}:
            return [axis for axis in ("x", "y", "z") if axis != fixed_axis]

        normal = self._slice_normal_hint(slice_object, component_axis=component_axis)
        if normal is None:
            return ["x", "y", "z"]

        axes = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        allowed = []
        for axis_name, axis_vector in axes.items():
            if abs(float(np.dot(normal, axis_vector))) < 0.99:
                allowed.append(axis_name)
        return allowed or ["x", "y", "z"]

    def _slice_normal_hint(self, slice_object, component_axis: str | None = None):
        params = dict(getattr(slice_object, "parameters", {}) or {})
        kind = str(params.get("kind") or "").lower()
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        if kind == "axis":
            return axis_map.get(str(params.get("axis") or "").lower())
        if kind == "orthogonal":
            return axis_map.get((component_axis or "").lower())
        if kind == "polyline":
            normal = np.asarray(params.get("normal", []), dtype=float).reshape(-1)
            if normal.size != 3:
                return None
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                return None
            return normal / norm
        if kind != "plane":
            return None

        normal = np.asarray(params.get("normal", []), dtype=float).reshape(-1)
        if normal.size != 3:
            return None
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            return None
        return normal / norm

    def _is_slice_like_object(self, scene_object) -> bool:
        if scene_object is None:
            return False
        object_type = str(getattr(scene_object, "object_type", "") or "").lower()
        if object_type == "slice":
            return True
        params = dict(getattr(scene_object, "parameters", {}) or {})
        kind = str(params.get("kind") or "").lower()
        return kind in {"axis", "orthogonal", "plane"}

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

    def import_well_log_data(self):
        dialog = WellLogImportDialog(self.import_service, self)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        payload = dialog.get_import_payload()
        if not payload:
            return
        file_path = str(payload["file_path"])
        import_spec = dict(payload["import_spec"])
        self._run_worker(
            "正在导入测井数据...",
            lambda: self.import_service.load_well_log_dataset(file_path, import_spec),
            self._on_well_log_import_finished,
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

    def _on_well_log_import_finished(self, dataset):
        reference_object = self._pick_well_reference_object()
        mapped_to_reference = False
        if (
            reference_object is not None
            and self._is_bounds_far_from_scene(dataset.bounds, [reference_object.bounds])
        ):
            mapped_to_reference = self._remap_dataset_points_to_bounds(dataset, reference_object.bounds)

        scene_object = self.scene_service.add_dataset(
            dataset,
            render=True,
            name=dataset.name,
            object_type=WELL_LOG_OBJECT_TYPE,
        )
        self.scene_manager.add_object(scene_object)
        self.on_object_selected(scene_object.object_id)
        self._fit_scene_to_objects()
        if mapped_to_reference and reference_object is not None:
            self.statusBar().showMessage(
                f"测井数据导入成功，已映射到 {reference_object.name} 的坐标范围。",
                5000,
            )
        else:
            self.statusBar().showMessage(f"测井数据导入成功：{scene_object.name}", 4000)
        self.plotter.render()

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

    def toggle_axis_scales(self):
        visible = self.axis_scale_component.toggle_visible()
        self.statusBar().showMessage(
            "坐标刻度已显示" if visible else "坐标刻度已隐藏",
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

    def _set_axes_visible(self, visible: bool):
        self._show_view_axes = bool(visible)
        self.view_axes.setVisible(self._show_view_axes)
        if hasattr(self, "toggle_axes_action"):
            self.toggle_axes_action.blockSignals(True)
            self.toggle_axes_action.setChecked(self._show_view_axes)
            self.toggle_axes_action.blockSignals(False)
        self.statusBar().showMessage("方向组件已显示" if self._show_view_axes else "方向组件已隐藏", 2000)

    def _set_axis_scales_visible(self, visible: bool):
        self.axis_scale_component.set_visible(bool(visible))
        if hasattr(self, "toggle_axis_scales_action"):
            self.toggle_axis_scales_action.blockSignals(True)
            self.toggle_axis_scales_action.setChecked(bool(visible))
            self.toggle_axis_scales_action.blockSignals(False)
        self.statusBar().showMessage("坐标刻度已显示" if visible else "坐标刻度已隐藏", 2000)

    def _set_selection_highlight_visible(self, visible: bool):
        self._show_selection_highlight = bool(visible)
        if hasattr(self, "toggle_selection_highlight_action"):
            self.toggle_selection_highlight_action.blockSignals(True)
            self.toggle_selection_highlight_action.setChecked(bool(visible))
            self.toggle_selection_highlight_action.blockSignals(False)
        if not visible:
            self.clear_selection_outline()
        else:
            scene_object = self._get_selected_scene_object()
            if scene_object is not None:
                source_object = (
                    self.scene_service.get_object(scene_object.source_object_id)
                    if scene_object.source_object_id
                    else scene_object
                )
                highlight_targets = [scene_object]
                if source_object is not None and source_object.object_id != scene_object.object_id:
                    highlight_targets.append(source_object)
                self._highlight_objects(highlight_targets)
        self.statusBar().showMessage("选中高亮已显示" if visible else "选中高亮已隐藏", 2000)

    def _set_scene_tree_visible(self, visible: bool):
        if not hasattr(self, "scene_dock"):
            return
        self.scene_dock.setVisible(bool(visible))
        if hasattr(self, "toggle_scene_tree_action"):
            self.toggle_scene_tree_action.blockSignals(True)
            self.toggle_scene_tree_action.setChecked(bool(visible))
            self.toggle_scene_tree_action.blockSignals(False)
        self.statusBar().showMessage("场景树已显示" if visible else "场景树已隐藏", 2000)

    def _on_scene_tree_dock_visibility_changed(self, visible: bool):
        if hasattr(self, "toggle_scene_tree_action"):
            self.toggle_scene_tree_action.blockSignals(True)
            self.toggle_scene_tree_action.setChecked(bool(visible))
            self.toggle_scene_tree_action.blockSignals(False)

    def _sync_view_menu_actions(self):
        if hasattr(self, "toggle_axes_action"):
            self.toggle_axes_action.blockSignals(True)
            self.toggle_axes_action.setChecked(self._show_view_axes)
            self.toggle_axes_action.blockSignals(False)
        if hasattr(self, "toggle_axis_scales_action"):
            self.toggle_axis_scales_action.blockSignals(True)
            self.toggle_axis_scales_action.setChecked(self.axis_scale_component.get_visible())
            self.toggle_axis_scales_action.blockSignals(False)
        if hasattr(self, "toggle_selection_highlight_action"):
            self.toggle_selection_highlight_action.blockSignals(True)
            self.toggle_selection_highlight_action.setChecked(self._show_selection_highlight)
            self.toggle_selection_highlight_action.blockSignals(False)
        if hasattr(self, "toggle_scene_tree_action"):
            self.toggle_scene_tree_action.blockSignals(True)
            self.toggle_scene_tree_action.setChecked(
                bool(getattr(self, "scene_dock", None) and (not self.scene_dock.isHidden()))
            )
            self.toggle_scene_tree_action.blockSignals(False)

    def on_object_selected(self, object_id: str):
        previous_selected_id = self.selected_object_id
        self.selected_object_id = object_id
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None:
            self._sync_scalar_bar_action(None)
            return
        if previous_selected_id and previous_selected_id != object_id and self.plotter.is_polyline_drawing():
            self.plotter.cancel_polyline_drawing()
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
        self._sync_scalar_bar_action(scene_object)
        self._sync_scalar_bar_owner_with_selection(scene_object)
        self.statusBar().showMessage(f"已选中：{scene_object.name}", 2500)

    def _highlight_objects(self, scene_objects):
        self.clear_selection_outline()
        if not self._show_selection_highlight:
            return
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
        self._focus_scene_bounds(bounds_array)

    def _focus_scene_bounds(self, bounds_array):
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
        self.plotter.reset_camera()

    def _is_bounds_far_from_scene(self, bounds, other_bounds_list) -> bool:
        if not other_bounds_list:
            return False
        current = np.asarray(bounds, dtype=float).reshape(-1)
        if current.size != 6:
            return False
        cx = float((current[0] + current[1]) * 0.5)
        cy = float((current[2] + current[3]) * 0.5)
        current_scale = max(float(current[1] - current[0]), float(current[3] - current[2]), 1.0)

        for other in other_bounds_list:
            candidate = np.asarray(other, dtype=float).reshape(-1)
            if candidate.size != 6:
                continue
            overlap_x = not (current[1] < candidate[0] or candidate[1] < current[0])
            overlap_y = not (current[3] < candidate[2] or candidate[3] < current[2])
            if overlap_x and overlap_y:
                return False

            ox = float((candidate[0] + candidate[1]) * 0.5)
            oy = float((candidate[2] + candidate[3]) * 0.5)
            distance = float(np.hypot(cx - ox, cy - oy))
            other_scale = max(float(candidate[1] - candidate[0]), float(candidate[3] - candidate[2]), 1.0)
            if distance <= max(current_scale, other_scale) * 5.0:
                return False
        return True

    def _pick_well_reference_object(self):
        selected = self._get_selected_scene_object()
        if selected is not None and not self._is_well_log_scene_object(selected):
            return selected

        dataset_candidates = []
        for scene_object in self.scene_service.all_objects():
            if scene_object.dataset is None:
                continue
            if self._is_well_log_scene_object(scene_object):
                continue
            if scene_object.object_type == "dataset":
                dataset_candidates.append(scene_object)
        if dataset_candidates:
            return dataset_candidates[0]
        return None

    def _is_well_log_scene_object(self, scene_object) -> bool:
        try:
            schema = dict(getattr(scene_object.dataset, "source_schema", {}) or {})
        except Exception:
            return False
        return str(schema.get("loader") or "").lower() == "well_log"

    def _remap_dataset_points_to_bounds(self, dataset, target_bounds) -> bool:
        data = getattr(dataset, "data", None)
        if data is None or not hasattr(data, "points"):
            return False
        points = np.asarray(data.points, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return False

        source_bounds = np.asarray(dataset.bounds, dtype=float).reshape(-1)
        target = np.asarray(target_bounds, dtype=float).reshape(-1)
        if source_bounds.size != 6 or target.size != 6:
            return False

        mapped = points.copy()
        for axis in range(3):
            src_min = float(min(source_bounds[axis * 2], source_bounds[axis * 2 + 1]))
            src_max = float(max(source_bounds[axis * 2], source_bounds[axis * 2 + 1]))
            dst_min = float(min(target[axis * 2], target[axis * 2 + 1]))
            dst_max = float(max(target[axis * 2], target[axis * 2 + 1]))
            src_span = src_max - src_min
            if abs(src_span) <= 1e-12:
                mapped[:, axis] = (dst_min + dst_max) * 0.5
            else:
                mapped[:, axis] = (mapped[:, axis] - src_min) * (dst_max - dst_min) / src_span + dst_min

        data.points = mapped
        return True

    def _get_selected_scene_object(self):
        if not self.selected_object_id:
            return None
        return self.scene_service.get_object(self.selected_object_id)

    def _toggle_selected_scalar_bar(self, checked: bool):
        scene_object = self._get_selected_scene_object()
        if scene_object is None:
            self._sync_scalar_bar_action(None)
            return
        self._apply_style_update(scene_object.object_id, show_scalar_bar=checked)

    def _sync_scalar_bar_action(self, scene_object):
        if not hasattr(self, "toggle_scalar_bar_action"):
            return
        enabled = scene_object is not None and scene_object.active_scalar is not None
        self.toggle_scalar_bar_action.blockSignals(True)
        self.toggle_scalar_bar_action.setEnabled(enabled)
        self.toggle_scalar_bar_action.setChecked(bool(enabled and scene_object.style.show_scalar_bar))
        self.toggle_scalar_bar_action.blockSignals(False)

    def _sync_scalar_bar_owner_with_selection(self, scene_object):
        render_manager = getattr(self.scene_service, "render_manager", None)
        if render_manager is None:
            return

        has_scalar = scene_object is not None and scene_object.active_scalar is not None
        wants_scalar_bar = bool(has_scalar and scene_object.style.show_scalar_bar and scene_object.visible)
        owner_id = getattr(render_manager, "_scalar_bar_owner_id", None)

        if wants_scalar_bar:
            if owner_id != scene_object.object_id:
                # 单色标模式：切换选中对象时把色标切到当前对象
                self.scene_service.rerender_object(scene_object.object_id)
            return

        if owner_id is not None:
            render_manager._clear_scalar_bars(self.plotter)
            render_manager._scalar_bar_owner_id = None
            self.plotter.render()

    def _apply_style_update(self, object_id: str, **updates):
        scene_object = self.scene_service.update_style(object_id, **updates)
        if scene_object is None:
            return
        self.scene_manager.refresh_object(scene_object)
        if self.selected_object_id == object_id:
            self.property_panel.set_scene_object(scene_object)
            self._sync_scalar_bar_action(scene_object)

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
            self._sync_scalar_bar_action(None)
            self.clear_selection_outline()
            self.plotter.cancel_polyline_drawing()

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
            self._sync_scalar_bar_action(None)
            self.plotter.cancel_polyline_drawing()
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

    def _start_polyline_drawing(self, object_id: str, params: dict):
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None:
            return
        draw_plane = str(params.get("draw_plane") or "xoy").lower()
        draw_value = float(params.get("draw_value", params.get("draw_z", 0.0)))
        self._polyline_owner = "slice"
        clip_bounds = np.asarray(scene_object.bounds, dtype=float) if scene_object.bounds is not None else None
        self.plotter.set_view(self._view_name_for_draw_plane(draw_plane))
        self.plotter.start_polyline_drawing(
            draw_value,
            clip_bounds=clip_bounds,
            draw_plane=draw_plane,
        )
        self.slice_panel.set_polyline_state(True, 0)
        self.clip_panel.set_mask_state(False, 0)
        self.statusBar().showMessage(f"已开始为 {scene_object.name} 绘制折线剖面", 3000)

    def _cancel_polyline_drawing(self):
        self._polyline_owner = None
        self.plotter.cancel_polyline_drawing()

    def _create_polyline_section(self, object_id: str, params: dict):
        points = self.plotter.get_polyline_points()
        if len(points) < 2:
            QMessageBox.information(self, "折线剖面", "请先在主视图中绘制至少两个折线点。")
            return
        draw_plane = str(params.get("draw_plane") or getattr(self.plotter, "get_polyline_draw_plane", lambda: "xoy")()).lower()
        self._run_worker(
            "正在生成折线剖面...",
            lambda: self.scene_service.create_polyline_section(
                object_id,
                points,
                top_z=params["top_z"],
                bottom_z=params["bottom_z"],
                draw_plane=draw_plane,
                line_step=params["line_step"],
                vertical_samples=params["vertical_samples"],
                render=False,
                add_to_scene=False,
            ),
            self._on_polyline_section_ready,
        )

    def _start_mask_drawing(self, object_id: str, params: dict):
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None:
            return
        draw_plane = str(params.get("draw_plane") or "xoy").lower()
        draw_value = float(params.get("draw_value", params.get("draw_z", 0.0)))
        self._polyline_owner = "mask"
        clip_bounds = np.asarray(scene_object.bounds, dtype=float) if scene_object.bounds is not None else None
        grid_spec = None
        if scene_object.dataset is not None and scene_object.dataset.is_regular_grid:
            grid_spec = {
                "origin": tuple(float(v) for v in scene_object.data.origin),
                "spacing": tuple(float(v) for v in scene_object.data.spacing),
                "dims": tuple(int(v) for v in scene_object.data.dimensions),
            }
        self.plotter.set_view(self._view_name_for_draw_plane(draw_plane))
        self.plotter.start_polyline_drawing(
            draw_value,
            clip_bounds=clip_bounds,
            draw_plane=draw_plane,
            snap_to_grid=grid_spec is not None,
            grid_spec=grid_spec,
            show_grid_overlay=grid_spec is not None,
        )
        self.clip_panel.set_mask_state(True, 0)
        self.statusBar().showMessage(f"已开始为 {scene_object.name} 绘制掩膜边界", 3000)

    def _create_mask_clip(self, object_id: str):
        points = self.plotter.get_polyline_points()
        if len(points) < 3:
            QMessageBox.information(self, "掩膜裁剪", "请先绘制至少三个边界点。")
            return
        draw_plane = getattr(self.plotter, "get_polyline_draw_plane", lambda: "xoy")()
        self._run_worker(
            "正在生成掩膜裁剪...",
            lambda: self.scene_service.create_mask_clip_from_polyline(
                object_id,
                points,
                draw_plane=draw_plane,
                render=False,
                add_to_scene=False,
            ),
            self._on_mask_clip_ready,
        )

    def _view_name_for_draw_plane(self, draw_plane: str) -> str:
        plane = str(draw_plane or "xoy").strip().lower()
        mapping = {
            "xoy": "top",
            "xoz": "back",
            "yoz": "left",
        }
        return mapping.get(plane, "top")

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

    def _create_grid_index_clip(self, object_id: str, index_bounds):
        if self.is_grid_index_pick_active():
            self.cancel_grid_index_pick(silent=True)
        self._run_worker(
            "正在按格点索引裁剪...",
            lambda: self.scene_service.create_grid_index_clip(
                object_id,
                index_bounds,
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def is_grid_index_pick_active(self) -> bool:
        return bool(self._grid_pick_active)

    def _start_grid_index_pick(self, object_id: str):
        scene_object = self.scene_service.get_object(object_id)
        if scene_object is None or scene_object.dataset is None or not scene_object.dataset.is_regular_grid:
            QMessageBox.information(self, "格点交互", "仅规则体数据支持交互选格点。")
            return

        self.cancel_grid_index_pick(silent=True)
        self._grid_pick_active = True
        self._grid_pick_object_id = object_id
        self._grid_pick_dims = np.asarray(scene_object.data.dimensions, dtype=int)
        self._grid_pick_origin = np.asarray(scene_object.data.origin, dtype=float)
        self._grid_pick_spacing = np.asarray(scene_object.data.spacing, dtype=float)
        self._grid_pick_anchor = None
        self._grid_pick_current = None
        self.plotter.set_view("top")
        self.clip_panel.set_grid_pick_state(True, "交互中：左键选两点，Enter确认，Esc取消")
        self.statusBar().showMessage("请在主视图左键点击两个格点角点。", 4000)

    def handle_grid_index_pick_click(self, screen_pos):
        if not self._grid_pick_active or self._grid_pick_object_id is None:
            return
        scene_object = self.scene_service.get_object(self._grid_pick_object_id)
        if scene_object is None:
            self.cancel_grid_index_pick()
            return

        z_ref = float(scene_object.bounds[5])
        world = CoordinateConverter.screen_to_horizontal_plane(
            self.plotter,
            screen_pos,
            z_ref,
            clip_to_bounds=False,
        )
        if world is None:
            self.statusBar().showMessage("当前视角无法投影到模型平面。", 2500)
            return

        grid_index = self._world_to_grid_index(np.asarray(world, dtype=float))
        if grid_index is None:
            return

        if self._grid_pick_anchor is None:
            self._grid_pick_anchor = grid_index
            self._grid_pick_current = grid_index.copy()
            self._update_grid_pick_preview()
            self.statusBar().showMessage("已记录第一点，请点击第二点。", 2500)
            return

        self._grid_pick_current = grid_index
        bounds = self._grid_index_bounds_from_pair(self._grid_pick_anchor, self._grid_pick_current)
        self.clip_panel.set_grid_index_values(bounds)
        self._update_grid_pick_preview()
        self.finish_grid_index_pick()

    def finish_grid_index_pick(self):
        if not self._grid_pick_active:
            return
        if self._grid_pick_anchor is None or self._grid_pick_current is None:
            self.statusBar().showMessage("格点范围尚未完成，请至少选择两点。", 2500)
            return
        bounds = self._grid_index_bounds_from_pair(self._grid_pick_anchor, self._grid_pick_current)
        self.clip_panel.set_grid_index_values(bounds)
        self.clip_panel.set_grid_pick_state(False, "已完成交互，可直接按格点裁剪")
        self.statusBar().showMessage("格点范围已回填。", 2500)
        self._grid_pick_active = False
        self._clear_grid_pick_preview()

    def cancel_grid_index_pick(self, silent: bool = False):
        was_active = self._grid_pick_active or self._grid_pick_anchor is not None
        self._grid_pick_active = False
        self._grid_pick_object_id = None
        self._grid_pick_dims = None
        self._grid_pick_origin = None
        self._grid_pick_spacing = None
        self._grid_pick_anchor = None
        self._grid_pick_current = None
        self._clear_grid_pick_preview()
        self.clip_panel.set_grid_pick_state(False, "未开始交互")
        if was_active and not silent:
            self.statusBar().showMessage("已取消格点交互。", 2500)

    def _world_to_grid_index(self, world: np.ndarray):
        if self._grid_pick_dims is None or self._grid_pick_origin is None or self._grid_pick_spacing is None:
            return None
        spacing = np.where(np.abs(self._grid_pick_spacing) < 1e-12, 1.0, self._grid_pick_spacing)
        raw = np.rint((world - self._grid_pick_origin) / spacing).astype(int)
        raw = np.clip(raw, 0, self._grid_pick_dims - 1)
        return raw

    def _grid_index_bounds_from_pair(self, a: np.ndarray, b: np.ndarray):
        low = np.minimum(a, b).astype(int)
        high = np.maximum(a, b).astype(int)
        return (int(low[0]), int(high[0]), int(low[1]), int(high[1]), int(low[2]), int(high[2]))

    def _grid_index_world_box_bounds(self, index_bounds):
        if self._grid_pick_origin is None or self._grid_pick_spacing is None:
            return None
        ix0, ix1, iy0, iy1, iz0, iz1 = [int(v) for v in index_bounds]
        xs = self._grid_pick_origin[0] + self._grid_pick_spacing[0] * np.array([ix0, ix1], dtype=float)
        ys = self._grid_pick_origin[1] + self._grid_pick_spacing[1] * np.array([iy0, iy1], dtype=float)
        zs = self._grid_pick_origin[2] + self._grid_pick_spacing[2] * np.array([iz0, iz1], dtype=float)
        return (
            float(min(xs)),
            float(max(xs)),
            float(min(ys)),
            float(max(ys)),
            float(min(zs)),
            float(max(zs)),
        )

    def _update_grid_pick_preview(self):
        self._clear_grid_pick_preview()
        if self._grid_pick_anchor is None or self._grid_pick_current is None:
            return
        world_bounds = self._grid_index_world_box_bounds(
            self._grid_index_bounds_from_pair(self._grid_pick_anchor, self._grid_pick_current)
        )
        if world_bounds is None:
            return
        box = pv.Box(bounds=world_bounds)
        self._grid_pick_preview_actor = self.plotter.add_mesh(
            box,
            style="wireframe",
            color="cyan",
            line_width=2,
            opacity=0.9,
            pickable=False,
            name="grid_pick_preview",
        )
        self.plotter.render()

    def _clear_grid_pick_preview(self):
        if self._grid_pick_preview_actor is None:
            return
        try:
            self.plotter.remove_actor(self._grid_pick_preview_actor)
        except Exception:
            pass
        self._grid_pick_preview_actor = None

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

    def run_drillhole_mapping(self):
        source_object = self._pick_drillhole_source_object()
        if source_object is None:
            QMessageBox.information(self, "钻孔映射", "请先导入并选中一个体数据对象。")
            return
        well_objects = [
            scene_object
            for scene_object in self.scene_service.all_objects()
            if scene_object.dataset is not None and self._is_well_log_scene_object(scene_object)
        ]
        if not well_objects:
            QMessageBox.information(self, "钻孔映射", "请先导入测井数据对象。")
            return

        well_object = well_objects[0]
        if len(well_objects) > 1:
            names = [scene_object.name for scene_object in well_objects]
            selected_name, ok = QInputDialog.getItem(
                self,
                "选择测井对象",
                "请选择用于钻孔映射的测井对象：",
                names,
                0,
                False,
            )
            if not ok:
                return
            for scene_object in well_objects:
                if scene_object.name == selected_name:
                    well_object = scene_object
                    break

        bounds = np.asarray(source_object.bounds, dtype=float)
        default_radius = max((bounds[1] - bounds[0]) * 0.015, 1.0)
        radius, ok = QInputDialog.getDouble(
            self,
            "钻孔映射",
            "井筒半径：",
            float(default_radius),
            0.001,
            1e9,
            3,
        )
        if not ok:
            return

        overlay_ids = [
            scene_object.object_id
            for scene_object in self.scene_service.all_objects()
            if scene_object.object_id not in {source_object.object_id, well_object.object_id}
            and scene_object.dataset is not None
            and not scene_object.dataset.is_regular_grid
            and not scene_object.dataset.is_point_set
            and scene_object.object_type in {"dataset", "isosurface"}
        ]
        self._run_worker(
            "正在执行钻孔映射...",
            lambda: self.scene_service.create_drillhole_mapping(
                source_object.object_id,
                well_object_id=well_object.object_id,
                overlay_object_ids=overlay_ids,
                radius=float(radius),
                render=False,
                add_to_scene=False,
            ),
            self._on_created_objects_ready,
        )

    def _pick_drillhole_source_object(self):
        selected = self._get_selected_scene_object()
        if (
            selected is not None
            and selected.dataset is not None
            and selected.object_type == "dataset"
            and not self._is_well_log_scene_object(selected)
        ):
            return selected
        for scene_object in self.scene_service.all_objects():
            if scene_object.dataset is None:
                continue
            if scene_object.object_type != "dataset":
                continue
            if self._is_well_log_scene_object(scene_object):
                continue
            if scene_object.dataset.is_point_set:
                continue
            return scene_object
        return None

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

    def _on_polyline_section_ready(self, result):
        self._on_created_objects_ready(result)
        self._polyline_owner = None
        self.plotter.cancel_polyline_drawing()

    def _on_mask_clip_ready(self, result):
        self._on_created_objects_ready(result)
        self._polyline_owner = None
        self.plotter.cancel_polyline_drawing()

    def _on_polyline_points_changed(self, point_count: int):
        if self._polyline_owner == "mask":
            self.clip_panel.set_mask_state(self.plotter.is_polyline_drawing(), point_count)
        else:
            self.slice_panel.set_polyline_state(self.plotter.is_polyline_drawing(), point_count)

    def _on_polyline_finished(self, points):
        if self._polyline_owner == "mask":
            self.clip_panel.set_mask_state(False, len(points))
        else:
            self.slice_panel.set_polyline_state(False, len(points))

    def _on_polyline_cancelled(self):
        self._polyline_owner = None
        self.slice_panel.set_polyline_state(False, 0)
        self.clip_panel.set_mask_state(False, 0)

    def _update_view_axes_position(self):
        if hasattr(self, "view_axes") and hasattr(self, "plotter"):
            plotter_size = self.plotter.size()
            axes_size = self.view_axes.size().width()
            margin = 10
            self.view_axes.move(plotter_size.width() - axes_size - margin, margin)
            self.view_axes.setVisible(self._show_view_axes)

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
