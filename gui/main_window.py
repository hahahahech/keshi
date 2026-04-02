"""
主窗口
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QDockWidget, QMainWindow, QMessageBox

from core.imported_model import ImportedModel
from gui.SceneManagerPanel import SceneManagerPanel
from gui.axis_scale_component import AxisScaleComponent
from gui.interactive_view import InteractiveView
from gui.professional_toolbar import ProfessionalToolbar
from gui.view_axes_2d import ViewAxes2D


class MainWindow(QMainWindow):
    """纯可视化主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维可视化软件")
        self.setGeometry(100, 100, 1600, 900)

        self._selection_outline_actor = None
        self.initial_workspace_bounds = np.array([0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0])

        self._create_menu_bar()
        self._create_status_bar()
        self._create_main_widget()

        self.statusBar().showMessage("就绪")
        if hasattr(self, "plotter") and hasattr(self.plotter, "status_message"):
            self.plotter.status_message.connect(self.statusBar().showMessage)

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        import_model_action = QAction("导入模型", self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        file_menu.addSeparator()
        export_screenshot_action = QAction("导出截图", self)
        export_screenshot_action.triggered.connect(self.export_screenshot)
        file_menu.addAction(export_screenshot_action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("视图")
        actions = [
            ("重置视图", self.reset_view),
            ("显示方向组件", self.toggle_axes),
            ("显示网格", self.toggle_grid),
            ("显示原点坐标轴", self.toggle_origin_axes),
            ("显示坐标轴刻度", self.toggle_axis_scales),
            ("清除选中轮廓", self.clear_section_outline),
        ]
        for text, callback in actions:
            action = QAction(text, self)
            action.triggered.connect(callback)
            view_menu.addAction(action)

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
            workspace_bounds=self.initial_workspace_bounds,
            background_color="white",
        )
        self.setCentralWidget(self.plotter)

        self.toolbar = ProfessionalToolbar(self)
        self.toolbar.set_parent_window(self)
        self.toolbar.action_triggered.connect(self._on_toolbar_action)
        self.addToolBar(self.toolbar)

        scene_dock = QDockWidget("场景管理", self)
        scene_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        scene_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self.scene_manager = SceneManagerPanel(self)
        self.scene_manager.set_plotter(self.plotter)
        self.scene_manager.opacityChanged.connect(self.on_opacity_changed)
        self.scene_manager.objectSelected.connect(self.on_object_selected)
        scene_dock.setWidget(self.scene_manager)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, scene_dock)
        self._scene_dock = scene_dock

        self.view_axes = ViewAxes2D(self.plotter, size=100)
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
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    self.view_axes.update_camera_direction(direction / direction_norm, view_up)
            except Exception:
                return

        self.plotter.view_changed.connect(update_view_axes)
        QTimer.singleShot(100, lambda: [update_view_axes(), self._update_view_axes_position()])

    def _on_toolbar_action(self, action_name: str):
        action_map = {
            "import_model": self.import_model,
            "reset_view": self.reset_view,
            "toggle_axes": self.toggle_axes,
            "toggle_grid": self.toggle_grid,
            "toggle_origin_axes": self.toggle_origin_axes,
            "toggle_axis_scales": self.toggle_axis_scales,
            "clear_section_outline": self.clear_section_outline,
            "export_screenshot": self.export_screenshot,
        }

        callback = action_map.get(action_name)
        if callback:
            callback()
        else:
            self.statusBar().showMessage(f"动作 {action_name} 未实现", 2000)

    def import_model(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入三维模型",
            "",
            (
                "三维模型 (*.vtk *.vtu *.ply *.obj *.stl);;"
                "VTK 文件 (*.vtk *.vtu);;"
                "PLY 文件 (*.ply);;"
                "OBJ 文件 (*.obj);;"
                "STL 文件 (*.stl);;"
                "所有文件 (*)"
            ),
        )
        if not file_paths:
            return

        imported = []
        failures = []
        for file_path in file_paths:
            try:
                mesh = pv.read(file_path)
                model = ImportedModel(mesh=mesh, file_path=file_path, name=Path(file_path).stem)
                model.create_actor(self.plotter)
                self.scene_manager.add_object(model, category="模型")
                imported.append(model)
            except Exception as exc:
                failures.append(f"{Path(file_path).name}: {exc}")

        if imported:
            self._fit_scene_to_models()
            self.plotter.reset_camera()
            self.plotter.render()

        if imported and not failures:
            self.statusBar().showMessage(f"成功导入 {len(imported)} 个模型", 3000)
            return

        if imported and failures:
            self.statusBar().showMessage(
                f"导入完成：成功 {len(imported)} 个，失败 {len(failures)} 个",
                4000,
            )
            QMessageBox.warning(self, "部分导入失败", "\n".join(failures))
            return

        QMessageBox.critical(self, "导入失败", "\n".join(failures) or "未能读取所选文件")

    def _fit_scene_to_models(self):
        models = self.scene_manager.get_objects_by_type("model")
        if not models:
            return

        bounds_array = []
        for model in models:
            if hasattr(model, "mesh") and model.mesh is not None:
                bounds_array.append(model.mesh.bounds)

        if not bounds_array:
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
            ]
        )
        self.plotter.set_workspace_bounds(workspace_bounds)

    def reset_view(self):
        self.plotter.set_workspace_bounds(self.initial_workspace_bounds.copy())
        self.plotter.reset_camera()
        self.statusBar().showMessage("视图已重置", 2000)

    def toggle_axes(self):
        if hasattr(self, "view_axes"):
            self.view_axes.setVisible(not self.view_axes.isVisible())
            status = "显示" if self.view_axes.isVisible() else "隐藏"
            self.statusBar().showMessage(f"方向组件已{status}", 2000)

    def toggle_origin_axes(self):
        if hasattr(self, "plotter"):
            self.plotter.toggle_origin_axes()
            status = "显示" if self.plotter.get_show_origin_axes() else "隐藏"
            self.statusBar().showMessage(f"原点坐标轴已{status}", 2000)

    def toggle_axis_scales(self):
        visible = self.axis_scale_component.toggle_visible()
        status = "显示" if visible else "隐藏"
        self.statusBar().showMessage(f"坐标轴刻度已{status}", 2000)

    def toggle_grid(self):
        if hasattr(self, "plotter"):
            self.plotter.toggle_grid()
            status = "显示" if self.plotter.get_show_grid() else "隐藏"
            self.statusBar().showMessage(f"网格已{status}", 2000)

    def clear_section_outline(self):
        if self._selection_outline_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_outline_actor)
            except Exception:
                pass
            self._selection_outline_actor = None
            self.plotter.render()
        self.statusBar().showMessage("选中轮廓已清除", 2000)

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

    def export_screenshot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前截图",
            "",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;BMP 图片 (*.bmp)",
        )
        if not file_path:
            return

        try:
            self.plotter.screenshot(file_path)
            self.statusBar().showMessage(f"截图已导出: {file_path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"导出截图时发生错误: {exc}")

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "三维可视化软件\n\n"
            "版本: 0.1.0\n\n"
            "功能：\n"
            "- 三维场景浏览\n"
            "- 图层可见性与透明度控制\n"
            "- 网格/坐标轴辅助显示\n"
            "- 当前视图截图导出",
        )

    def on_opacity_changed(self, obj_id: str, opacity: int):
        self.scene_manager.on_scene_opacity_changed(obj_id, opacity)

    def on_object_selected(self, obj_id: str):
        self.clear_section_outline()

        try:
            item = self.scene_manager.get_item_by_id(obj_id)
            if not item or not item.data_object:
                self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
                return

            actors = []
            data_object = item.data_object
            if hasattr(data_object, "actor") and data_object.actor:
                actors = [data_object.actor]
            elif hasattr(data_object, "actors") and data_object.actors:
                actors = list(data_object.actors)

            if not actors:
                self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
                return

            actor = actors[0]
            mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
            polydata = mapper.GetInput() if mapper and hasattr(mapper, "GetInput") else None
            if polydata is None:
                self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
                return

            mesh = pv.wrap(polydata)
            if mesh.n_points == 0:
                self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
                return

            surface_mesh = mesh.extract_surface() if hasattr(mesh, "extract_surface") and not isinstance(mesh, pv.PolyData) else mesh
            edges = surface_mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
            )
            if edges.n_points == 0:
                self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
                return

            self._selection_outline_actor = self.plotter.add_mesh(
                edges,
                color="yellow",
                line_width=3,
                name=f"selection_outline_{obj_id}",
            )
            self.plotter.render()
            self.statusBar().showMessage(f"已选中对象: {obj_id}", 2000)
        except Exception as exc:
            self.statusBar().showMessage(f"对象选中处理失败: {exc}", 3000)
