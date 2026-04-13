"""
交互式建模视图核心类
"""
from PyQt6.QtWidgets import QLabel, QToolButton, QMenu, QWidgetAction, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import QFont, QIcon, QPixmap, QAction
import os
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np
from typing import Optional
from .camera import CameraController
from .coordinates import CoordinateConverter
from .events import EventHandler
# edit_mode 模块已移除，不再需要这些导入
# from ..vertical_axis import VerticalAxisController


class InteractiveView(QtInteractor):
    """交互式建模视图 - 实现轨道摄像机控制"""
    
    # 信号定义
    view_changed = pyqtSignal()  # 视图改变时发出信号
    status_message = pyqtSignal(str)  # 状态消息信号
    mode_changed = pyqtSignal(str)  # 模式改变时发出信号，参数是模式名称
    tool_changed = pyqtSignal(str)  # 工具改变时发出信号，参数是工具名称
    vertex_selected = pyqtSignal(int, object)  # 顶点选中信号 (vertex_id, world_position)
    polyline_changed = pyqtSignal(int)  # 折线绘制点数变化
    polyline_finished = pyqtSignal(object)  # 折线绘制完成
    polyline_cancelled = pyqtSignal()  # 折线绘制取消
    
    def __init__(self, parent=None, 
                 workspace_bounds: Optional[np.ndarray] = None,
                 background_color: str = 'white'):
        """初始化交互式视图"""
        super().__init__(parent)
        
        # 保存父窗口引用
        self.parent_window = parent
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 设置背景颜色
        self.set_background(background_color)
        
        # 建模空间边界
        if workspace_bounds is None:
            self.workspace_bounds = self._get_default_workspace_bounds()
        else:
            self.workspace_bounds = np.array(workspace_bounds, dtype=np.float64)
        
        # 轨道摄像机参数
        self._orbit_center = self._calculate_workspace_center()
        self._camera_distance = self._calculate_initial_distance()
        
        # 投影模式：True=正交投影，False=透视投影
        self._is_orthographic = False
        
        # 显示状态
        self._show_grid = False  # 是否显示网格
        self._show_origin_axes = False  # 是否显示原点坐标轴
        self._grid_actor = None  # 网格actor
        self._origin_axes_actor = None  # 原点坐标轴actor
        self._grid_spacing = 10.0  # 网格间距
        
        # 模式选择
        self._current_mode = 'object'  # 当前模式：'object'（物体模式）或 'edit'（编辑模式）
        
        # 编辑工具选择
        self._current_tool = None  # 当前工具：'point', 'line', 'curve', 'plane' 或 None
        self._tool_buttons = {}  # 存储工具按钮引用
        
        # 物体模式操作工具选择
        self._current_object_tool = None  # 当前物体操作工具：'select', 'box_select', 'translate', 'scale', 'rotate' 或 None
        self._object_tool_buttons = {}  # 存储物体操作工具按钮引用
        
        # 鼠标交互状态
        self._last_mouse_pos = None
        self._is_rotating = False
        self._is_panning = False
        self._is_zooming = False

        # 折线剖面绘制状态
        self._polyline_drawing = False
        self._polyline_draw_z = 0.0
        self._polyline_clip_bounds: Optional[np.ndarray] = None
        self._polyline_points: list[tuple[float, float, float]] = []
        self._polyline_hover_point: Optional[np.ndarray] = None
        self._polyline_line_actor = None
        self._polyline_points_actor = None
        self._polyline_preview_actor = None
        
        # 初始化摄像机
        CameraController.setup_camera(self)
        
        
        # 初始化网格和坐标轴（默认不显示）
        self._update_grid()
        self._update_origin_axes()
        # 保留这些状态变量（用于向后兼容）
        self._current_mode = 'observer'
        self._current_tool = None
        self._current_object_tool = None
        

        # 创建垂直坐标轴控制器
        # self._vertical_axis_controller = VerticalAxisController(self)
        # self._vertical_axis_controller.setup_axis()
        # self._show_vertical_axis = True  # 垂直坐标轴显示状态
        self._show_vertical_axis = False  # 垂直坐标轴已禁用

        # 被屏幕拾取的点高亮状态缓存 (用于视觉反馈)
        self._picked_point_prev = None  # (point_id, color)

        # 初始化边界几何（不可操作，仅可选）
        self._init_boundary_geometry()
        
    # ========== 工作空间相关方法 ==========
    
    def _calculate_workspace_center(self) -> np.ndarray:
        """计算建模空间中心点"""
        bounds = self.workspace_bounds
        return np.array([
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0
        ])
    
    def _calculate_initial_distance(self) -> float:
        """计算初始摄像机距离"""
        bounds = self.workspace_bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
        return diagonal * 1.5

    def _get_default_workspace_bounds(self) -> np.ndarray:
        """获取默认工作空间边界"""
        return np.array([-100.0, 100.0, -100.0, 100.0, -50.0, 0.0], dtype=np.float64)

    def _create_workspace_bounds_mesh(self, bounds: np.ndarray) -> pv.PolyData:
        """创建工作空间边界框网格"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])

        lines_array = []
        for edge in edges:
            lines_array.extend([2, int(edge[0]), int(edge[1])])

        mesh = pv.PolyData(vertices)
        mesh.lines = np.array(lines_array, dtype=np.int32)
        return mesh

    def _create_grid_mesh(self, bounds: np.ndarray, grid_spacing: float = 10.0, z: float = 0.0) -> pv.PolyData:
        """创建参考网格"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]

        vertices = []
        lines_array = []

        x_values = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        for x in x_values:
            x = min(x, x_max)
            start_idx = len(vertices)
            vertices.append([x, y_min, z])
            vertices.append([x, y_max, z])
            lines_array.extend([2, start_idx, start_idx + 1])

        y_values = np.arange(y_min, y_max + grid_spacing, grid_spacing)
        for y in y_values:
            y = min(y, y_max)
            start_idx = len(vertices)
            vertices.append([x_min, y, z])
            vertices.append([x_max, y, z])
            lines_array.extend([2, start_idx, start_idx + 1])

        mesh = pv.PolyData(np.array(vertices))
        mesh.lines = np.array(lines_array, dtype=np.int32)
        return mesh

    def _create_origin_axes_mesh(self, bounds: np.ndarray) -> pv.PolyData:
        """创建原点坐标轴辅助网格"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        axis_length = min(x_max - x_min, y_max - y_min) * 0.6

        vertices = np.array([
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
        ])
        mesh = pv.PolyData(vertices)
        mesh.lines = np.array([2, 0, 1, 2, 0, 2], dtype=np.int32)
        return mesh

    def _create_basement_surface_mesh(self, bounds: np.ndarray, resolution: int = 20) -> pv.PolyData:
        """创建底面平面网格"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min = bounds[4]

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size_x = x_max - x_min
        size_y = y_max - y_min

        return pv.Plane(
            center=(center_x, center_y, z_min),
            direction=(0, 0, 1),
            i_size=size_x,
            j_size=size_y,
            i_resolution=resolution - 1,
            j_resolution=resolution - 1
        )
    
    def set_workspace_bounds(self, bounds: np.ndarray):
        """
        设置工作空间边界
        
        Parameters:
        -----------
        bounds : np.ndarray
            新的边界 [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        self.workspace_bounds = np.array(bounds, dtype=np.float64)
        
        # 重新计算轨道中心
        self._orbit_center = self._calculate_workspace_center()
        
        # 重新计算初始距离
        initial_distance = self._calculate_initial_distance()
        
        # 如果当前距离小于新的初始距离，则更新
        camera = self.renderer.GetActiveCamera()
        center = np.array(camera.GetFocalPoint())
        position = np.array(camera.GetPosition())
        current_distance = np.linalg.norm(position - center)
        
        if current_distance < initial_distance:
            self._camera_distance = initial_distance
            # 更新摄像机位置
            direction = position - center
            if np.linalg.norm(direction) > 1e-6:
                direction_normalized = direction / np.linalg.norm(direction)
                new_position = self._orbit_center + direction_normalized * initial_distance
                camera.SetPosition(new_position)
                camera.SetFocalPoint(self._orbit_center)
        
        # 移除旧的边界框（如果存在）
        if hasattr(self, '_workspace_bounds_actor'):
            for actor in self._workspace_bounds_actor:
                try:
                    self.remove_actor(actor)
                except:
                    pass
            self._workspace_bounds_actor = []
        
        # 更新网格和坐标轴（如果已显示）
        if self._show_grid:
            self._update_grid()
        if self._show_origin_axes:
            self._update_origin_axes()
            
        # 更新垂直坐标轴
        # if hasattr(self, '_vertical_axis_controller'):
        #     self._vertical_axis_controller.update_from_workspace(self.workspace_bounds)
        
        self.render()
        self.view_changed.emit()

    def _init_boundary_geometry(self):
        """初始化边界点/线/面为锁定对象（仅可选不可操作）"""
        bounds = self.workspace_bounds
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]

        # 8 个顶点
        corners = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])
        # 边界点创建已移除

        # 12 条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
        ]
        # 边界线创建已移除

        # 6 个面（保持透明，可选不可编辑）
        # 顶点顺序按右手坐标系设置，确保法向量指向空间外部
        faces = [
            [0, 3, 2, 1],  # bottom z_min, 法向量指向-Z（向下）
            [4, 5, 6, 7],  # top z_max, 法向量指向+Z（向上）
            [0, 1, 5, 4],  # front y_min, 法向量指向-Y（向前）
            [1, 2, 6, 5],  # right x_max, 法向量指向+X（向右）
            [2, 3, 7, 6],  # back y_max, 法向量指向+Y（向后）
            [3, 0, 4, 7],  # left x_min, 法向量指向-X（向左）
        ]
        # 边界面创建已移除
    
    def get_workspace_bounds(self) -> np.ndarray:
        """
        获取当前工作空间边界
        
        Returns:
        --------
        np.ndarray
            边界 [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        return self.workspace_bounds.copy()
    
    def _draw_workspace_bounds(self):
        """绘制建模空间边界框"""
        bounds = self.workspace_bounds
        
        # 创建边界框网格
        lines_mesh = self._create_workspace_bounds_mesh(bounds)
        
        # 添加到场景（使用淡灰色，半透明）
        actor = self.add_mesh(
            lines_mesh,
            color='black',
            line_width=1.0,
            opacity=1.0,
            name='workspace_bounds'
        )
        # 边界框仅用于视觉参考，禁止拾取，避免阻挡编辑点选/拖拽
        try:
            actor.PickableOff()
        except Exception:
            try:
                actor.SetPickable(False)
            except Exception:
                pass
        # 存储actor引用以便后续移除
        if not hasattr(self, '_workspace_bounds_actor'):
            self._workspace_bounds_actor = []
        self._workspace_bounds_actor.append(actor)
    
    def _draw_basement_surface(self):
        """绘制底面黑褐色曲面"""
        bounds = self.workspace_bounds
        
        # 创建底面曲面网格（低分辨率以提高性能）
        basement_mesh = self._create_basement_surface_mesh(bounds, resolution=20)
        
        # 添加到场景（黑褐色，不透明，平滑无边缘）
        # 黑褐色 RGB: (0.2, 0.15, 0.1) - 深褐色带黑色调
        actor = self.add_mesh(
            basement_mesh,
            color=(0.2, 0.15, 0.1),  # 黑褐色
            opacity=1.0,
            show_edges=False,  # 不显示边缘线
            smooth_shading=True,  # 平滑着色
            name='basement_surface'
        )
        
        # 底面仅用于视觉参考，禁止拾取
        try:
            actor.PickableOff()
        except Exception:
            try:
                actor.SetPickable(False)
            except Exception:
                pass
        
        # 存储actor引用
        if not hasattr(self, '_basement_surface_actor'):
            self._basement_surface_actor = None
        self._basement_surface_actor = actor
    
    # ========== 投影模式控制 ==========
    
    def set_projection_mode(self, orthographic: bool):
        """
        设置投影模式
        
        Parameters:
        -----------
        orthographic : bool
            True=正交投影，False=透视投影
        """
        self._is_orthographic = orthographic
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(orthographic)
        self.render()
        self.view_changed.emit()
    
    def get_projection_mode(self) -> bool:
        """
        获取当前投影模式
        
        Returns:
        --------
        bool
            True=正交投影，False=透视投影
        """
        return self._is_orthographic
    
    def toggle_projection_mode(self):
        """切换投影模式"""
        self.set_projection_mode(not self._is_orthographic)
    
    # ========== 快速视角切换 ==========
    
    def set_view(self, view_name: str):
        """设置快速视角"""
        legacy_view_name_map = {
            "top": "bottom",
            "bottom": "top",
        }
        CameraController.set_view(self, legacy_view_name_map.get(view_name, view_name))
    
    def reset_camera(self):
        """重置摄像机到初始位置 - 使用新的安全方法"""
        CameraController.reset_view_to_initial(self)
        self.view_changed.emit()
    
    def reset_camera_safe(self):
        """安全重置视图 - 新的专用方法"""
        CameraController.reset_view_to_initial(self)
    
    # ========== 网格和坐标轴控制 ==========
    
    def set_show_grid(self, show: bool):
        """设置是否显示网格"""
        self._show_grid = show
        self._update_grid()
        self.render()
        self.view_changed.emit()
    
    def get_show_grid(self) -> bool:
        """获取网格显示状态"""
        return self._show_grid
    
    def toggle_grid(self):
        """切换网格显示状态"""
        self.set_show_grid(not self._show_grid)
    
    def set_grid_spacing(self, spacing: float):
        """设置网格间距"""
        if spacing <= 0:
            raise ValueError("网格间距必须大于0")
        self._grid_spacing = spacing
        if self._show_grid:
            self._update_grid()
            self.render()
            self.view_changed.emit()
    
    def get_grid_spacing(self) -> float:
        """获取网格间距"""
        return self._grid_spacing
    
    def _update_grid(self):
        """更新网格显示"""

        if self._grid_actor is not None:
            try:
                self.remove_actor(self._grid_actor)
            except:
                pass
            self._grid_actor = None
        # 如果显示网格，创建新的网格
        if self._show_grid:
            grid_mesh = self._create_grid_mesh(self.workspace_bounds, self._grid_spacing, z=0.0)
            self._grid_actor = self.add_mesh(
                grid_mesh,
                color='lightgray',
                line_width=0.5,
                opacity=0.5,
                name='grid'
            )
            # 网格只作参考，禁用拾取
            try:
                self._grid_actor.PickableOff()
            except Exception:
                try:
                    self._grid_actor.SetPickable(False)
                except Exception:
                    pass
    
    def set_show_origin_axes(self, show: bool):
        """设置是否显示原点坐标轴"""
        self._show_origin_axes = show
        self._update_origin_axes()
        self.render()
        self.view_changed.emit()
    
    def get_show_origin_axes(self) -> bool:
        """获取原点坐标轴显示状态"""
        return self._show_origin_axes
    
    def toggle_origin_axes(self):
        """切换原点坐标轴显示状态"""
        self.set_show_origin_axes(not self._show_origin_axes)
    
    def toggle_vertical_axis(self):
        """切换垂直坐标轴显示状态"""
        # self.set_show_vertical_axis(not self._show_vertical_axis)
        pass  # 垂直坐标轴已禁用
    
    def set_show_vertical_axis(self, show: bool):
        """设置垂直坐标轴显示状态"""
        # self._show_vertical_axis = show
        # if hasattr(self, '_vertical_axis_controller'):
        #     self._vertical_axis_controller.set_visible(show)
        # self.render()
        pass  # 垂直坐标轴已禁用
    
    def get_show_vertical_axis(self):
        """获取垂直坐标轴显示状态"""
        # return self._show_vertical_axis
        return False  # 垂直坐标轴已禁
    def _update_origin_axes(self):
        """更新原点坐标轴显示"""
        # 移除旧的坐标轴
        if self._origin_axes_actor is not None:
            try:
                # 如果是列表，分别移除每个actor
                if isinstance(self._origin_axes_actor, list):
                    for actor in self._origin_axes_actor:
                        try:
                            self.remove_actor(actor)
                        except:
                            pass
                else:
                    self.remove_actor(self._origin_axes_actor)
            except:
                pass
            self._origin_axes_actor = None
        
        # 如果显示坐标轴，创建新的坐标轴
        if self._show_origin_axes:
            axes_mesh = self._create_origin_axes_mesh(self.workspace_bounds)
            # X轴用红色，Y轴用绿色
            # 由于PolyData不支持不同颜色，我们分别创建两个actor
            # X轴
            x_axis_vertices = np.array([
                [0.0, 0.0, 0.0],
                axes_mesh.points[1]  # X轴端点
            ])
            x_axis_mesh = pv.PolyData(x_axis_vertices)
            x_axis_mesh.lines = np.array([2, 0, 1], dtype=np.int32)
            
            # Y轴
            y_axis_vertices = np.array([
                [0.0, 0.0, 0.0],
                axes_mesh.points[2]  # Y轴端点
            ])
            y_axis_mesh = pv.PolyData(y_axis_vertices)
            y_axis_mesh.lines = np.array([2, 0, 1], dtype=np.int32)
            
            # 添加X轴（红色）
            x_actor = self.add_mesh(
                x_axis_mesh,
                color='red',
                line_width=4.0,  # 增大线条宽度
                name='origin_axis_x'
            )
            try:
                x_actor.PickableOff()
            except Exception:
                try:
                    x_actor.SetPickable(False)
                except Exception:
                    pass
            
            # 添加Y轴（绿色）
            y_actor = self.add_mesh(
                y_axis_mesh,
                color='green',
                line_width=4.0,  # 增大线条宽度
                name='origin_axis_y'
            )
            try:
                y_actor.PickableOff()
            except Exception:
                try:
                    y_actor.SetPickable(False)
                except Exception:
                    pass
            
            # 存储两个actor（使用列表）
            self._origin_axes_actor = [x_actor, y_actor]
    def undo(self):
        """撤销功能已移除"""
        return True
    
        return False

    def redo(self):
        """重做功能已移除"""
        return False

    def pick_point_at_screen(self, screen_pos: QPoint, pixel_threshold: int = 10) -> Optional[str]:
        """点选择功能已移除"""
        return None

    # ========== 折线剖面绘制 ==========

    def is_polyline_drawing(self) -> bool:
        return self._polyline_drawing

    def get_polyline_draw_z(self) -> float:
        return float(self._polyline_draw_z)

    def get_polyline_points(self) -> list[tuple[float, float, float]]:
        return [tuple(float(value) for value in point) for point in self._polyline_points]

    def start_polyline_drawing(self, z_value: float, clip_bounds=None):
        self._clear_polyline_actors()
        self._polyline_drawing = True
        self._polyline_draw_z = float(z_value)
        if clip_bounds is None:
            clip_bounds = self.workspace_bounds
        clip_bounds_array = np.asarray(clip_bounds, dtype=float).reshape(-1)
        if clip_bounds_array.size != 6:
            raise ValueError("折线绘制边界必须包含 6 个数值。")
        self._polyline_clip_bounds = clip_bounds_array.copy()
        self._polyline_points = []
        self._polyline_hover_point = None
        self.polyline_changed.emit(0)
        if hasattr(self, "status_message"):
            self.status_message.emit("进入折线绘制：左键加点，右键撤销，双击完成，Esc 取消")
        self.render()

    def finish_polyline_drawing(self):
        if len(self._polyline_points) < 2:
            raise ValueError("折线至少需要两个点。")
        self._polyline_drawing = False
        self._polyline_clip_bounds = None
        self._polyline_hover_point = None
        self._update_polyline_actors()
        self.polyline_finished.emit(self.get_polyline_points())
        if hasattr(self, "status_message"):
            self.status_message.emit(f"折线绘制完成，共 {len(self._polyline_points)} 个点")
        self.render()

    def cancel_polyline_drawing(self):
        was_active = self._polyline_drawing or bool(self._polyline_points)
        self._polyline_drawing = False
        self._polyline_clip_bounds = None
        self._polyline_points = []
        self._polyline_hover_point = None
        self._clear_polyline_actors()
        self.polyline_changed.emit(0)
        if was_active:
            self.polyline_cancelled.emit()
            if hasattr(self, "status_message"):
                self.status_message.emit("已取消折线绘制")
        self.render()

    def add_polyline_point(self, point):
        point_array = np.asarray(point, dtype=float).reshape(-1)
        if point_array.size == 2:
            point_array = np.array([point_array[0], point_array[1], self._polyline_draw_z], dtype=float)
        if point_array.size != 3:
            raise ValueError("折线点必须是二维或三维坐标。")
        point_array = self._clamp_polyline_point(point_array)
        point_tuple = tuple(float(value) for value in point_array.tolist())
        if self._polyline_points:
            previous = np.asarray(self._polyline_points[-1], dtype=float)
            if np.linalg.norm(previous - point_array) <= 1e-6:
                return
        self._polyline_points.append(point_tuple)
        self._update_polyline_actors()
        self.polyline_changed.emit(len(self._polyline_points))
        if hasattr(self, "status_message"):
            self.status_message.emit(f"已添加折线点 {len(self._polyline_points)}")
        self.render()

    def pop_polyline_point(self):
        if not self._polyline_points:
            return
        self._polyline_points.pop()
        self._update_polyline_actors()
        self.polyline_changed.emit(len(self._polyline_points))
        if hasattr(self, "status_message"):
            self.status_message.emit(f"已撤销，当前 {len(self._polyline_points)} 个点")
        self.render()

    def handle_polyline_click(self, screen_pos: QPoint):
        point = self._project_polyline_point(screen_pos)
        if point is None:
            if hasattr(self, "status_message"):
                self.status_message.emit("当前视角无法投影到折线绘制平面，请调整视角后重试")
            return
        self.add_polyline_point(point)

    def handle_polyline_hover(self, screen_pos: QPoint):
        if not self._polyline_drawing:
            return
        self._polyline_hover_point = self._project_polyline_point(screen_pos)
        self._update_polyline_actors()
        self.render()

    def _project_polyline_point(self, screen_pos: QPoint):
        point = CoordinateConverter.screen_to_horizontal_plane(
            self,
            screen_pos,
            self._polyline_draw_z,
            clip_to_bounds=False,
        )
        if point is None:
            return None
        return self._clamp_polyline_point(point)

    def _clamp_polyline_point(self, point):
        point_array = np.asarray(point, dtype=float).reshape(-1).copy()
        if point_array.size != 3:
            raise ValueError("折线点必须是三维坐标。")
        bounds = self._polyline_clip_bounds if self._polyline_clip_bounds is not None else self.workspace_bounds
        point_array[0] = np.clip(point_array[0], bounds[0], bounds[1])
        point_array[1] = np.clip(point_array[1], bounds[2], bounds[3])
        point_array[2] = self._polyline_draw_z
        return point_array

    def _clear_polyline_actors(self):
        for actor_name in ("_polyline_line_actor", "_polyline_points_actor", "_polyline_preview_actor"):
            actor = getattr(self, actor_name, None)
            if actor is None:
                continue
            try:
                self.remove_actor(actor)
            except Exception:
                pass
            setattr(self, actor_name, None)

    def _update_polyline_actors(self):
        self._clear_polyline_actors()
        if self._polyline_points:
            points_array = np.asarray(self._polyline_points, dtype=float)
            points_mesh = pv.PolyData(points_array)
            self._polyline_points_actor = self.add_mesh(
                points_mesh,
                color="orange",
                point_size=10,
                render_points_as_spheres=True,
                pickable=False,
                name="polyline_points_preview",
            )
            if len(points_array) >= 2:
                line_mesh = self._build_polyline_mesh(points_array)
                self._polyline_line_actor = self.add_mesh(
                    line_mesh,
                    color="orange",
                    line_width=3,
                    pickable=False,
                    name="polyline_line_preview",
                )
        if self._polyline_drawing and self._polyline_hover_point is not None and self._polyline_points:
            preview_points = np.vstack([
                np.asarray(self._polyline_points[-1], dtype=float),
                np.asarray(self._polyline_hover_point, dtype=float),
            ])
            preview_mesh = self._build_polyline_mesh(preview_points)
            self._polyline_preview_actor = self.add_mesh(
                preview_mesh,
                color="gold",
                line_width=2,
                opacity=0.7,
                pickable=False,
                name="polyline_hover_preview",
            )

    def _build_polyline_mesh(self, points: np.ndarray) -> pv.PolyData:
        mesh = pv.PolyData(points)
        mesh.lines = np.concatenate([[len(points)], np.arange(len(points), dtype=np.int32)]).astype(np.int32)
        return mesh
    
    # ========== 摄像机控制公共 API ==========
    
    def get_camera_info(self) -> dict:
        """获取当前摄像机信息"""
        return CameraController.get_camera_info(self)
    
    def set_camera_info(self, camera_info: dict):
        """设置摄像机信息"""
        CameraController.set_camera_info(self, camera_info)
    
    # ========== Qt 事件处理方法 ==========
    # 这些方法必须保留，因为它们是 Qt 的事件处理接口
    # 内部直接调用 EventHandler 的静态方法

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        EventHandler.mouse_press_event(self, event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        EventHandler.mouse_move_event(self, event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        EventHandler.mouse_release_event(self, event)
    
    def wheelEvent(self, event):
        """滚轮事件（缩放）"""
        EventHandler.wheel_event(self, event)
    
    def keyPressEvent(self, event):
        """键盘事件处理"""
        EventHandler.key_press_event(self, event)

    def contextMenuEvent(self, event):
        """右键菜单事件"""
        EventHandler.context_menu_event(self, event)

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event):
        """双击拾取并显示世界坐标。"""
        if self.is_polyline_drawing():
            try:
                self.finish_polyline_drawing()
            except Exception:
                pass
            event.accept()
            return
        try:
            point = self.pick_mouse_position()
            if point is not None and hasattr(self, "status_message"):
                x, y, z = [float(value) for value in point]
                self.status_message.emit(f"拾取坐标：X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)
    
    def clear_vertex_highlight(self):
        """清除顶点高亮"""
        try:
            if hasattr(self, '_vertex_picker'):
                self._vertex_picker.clear_highlight()
        except Exception as e:
            print(f"清除顶点高亮失败: {e}")
