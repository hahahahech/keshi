"""
摄像机控制相关方法
"""
from PyQt6.QtCore import QPoint
import numpy as np


class CameraController:
    """摄像机控制器 - 处理旋转、平移、缩放等操作"""
    
    @staticmethod
    def reset_view_to_initial(view):
        """重置视图：重新计算边界，并复用 setup_camera"""
        
        # 1. 计算工作空间中心 (找回模型中心)
        bounds = view.workspace_bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
        
        # 2. 计算合适的距离 (使用与初始化相同的计算方法)
        # 【关键修复】使用对角线计算，与初始化保持一致
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
        distance = diagonal * 1.5  # 距离设为对角线的1.5倍 
        
        # 3. 【关键】将计算结果更新到 view 对象中
        view._orbit_center = center
        view._camera_distance = distance
        
        # 4. 【关键】调用 setup_camera 应用这些值
        # 这样就保证了"初始化"和"重置"使用的是完全同一套摄像机参数设置逻辑
        CameraController.setup_camera(view)
    
    @staticmethod
    def setup_camera(view):
        """设置轨道摄像机"""
        # 设置摄像机位置（从斜上方看向中心）
        camera = view.renderer.GetActiveCamera()
        
        # 计算初始摄像机位置（等距投影）
        center = view._orbit_center
        distance = view._camera_distance
        
        # 默认视角：从斜上方看向中心（有角度的俯视）
        direction = np.array([-1.0, -1.0, 0.5])
        direction = direction / np.linalg.norm(direction)
        
        camera_pos = center + direction * distance
        
        camera.SetPosition(camera_pos)
        camera.SetFocalPoint(center)
        camera.SetViewUp(0, 0, 1)  # Z轴向上  
        
        # 设置投影模式
        camera.SetParallelProjection(view._is_orthographic)
        
        # 【关键修复】动态设置裁剪范围，确保模型完全可见
        # 近裁剪：避免摄像机太近时裁剪掉模型
        near_clip = distance * 0.001  # 距离的千分之一
        # 远裁剪：确保远处的模型完全可见，设置为摄像机距离的3倍
        far_clip = distance * 3.0
        camera.SetClippingRange(near_clip, far_clip)
        
        view.render()
    
    @staticmethod
    def handle_rotation(view, delta: QPoint):
        """处理旋转操作 - 使用球面坐标系"""
        camera = view.renderer.GetActiveCamera()
        
        # 获取当前摄像机参数
        center = np.array(camera.GetFocalPoint())
        position = np.array(camera.GetPosition())
        view_up = np.array(camera.GetViewUp())
        
        # 计算从中心到相机的方向向量
        direction = position - center
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return  # 避免除零错误
        
        # 归一化方向向量
        direction_normalized = direction / distance
        
        # ========== 使用球面坐标系计算当前角度 ==========
        # 计算当前方位角（在XY平面上的角度，0-2π）
        # azimuth = atan2(y, x)
        current_azimuth = np.arctan2(direction[1], direction[0])
        
        # 计算当前仰角（与XY平面的夹角，-π/2到π/2）
        # elevation = arcsin(z / distance)
        current_elevation = np.arcsin(np.clip(direction[2] / distance, -1.0, 1.0))
        
        # ========== 计算旋转增量 ==========
        rotation_sensitivity = 0.5  # 旋转灵敏度（度/像素）
        azimuth_delta = -delta.x() * rotation_sensitivity  # 水平旋转（左右：向右拖相机向右转）
        elevation_delta = delta.y() * rotation_sensitivity  # 垂直旋转（上下：向上拖相机向上看，注意屏幕Y向下）
        
        # ========== 应用旋转 ==========
        # 更新方位角
        new_azimuth = current_azimuth + np.radians(azimuth_delta)
        
        # 更新仰角（限制在-85°到85°之间，避免翻转）
        max_elevation = np.radians(85)
        new_elevation = np.clip(
            current_elevation + np.radians(elevation_delta),
            -max_elevation,
            max_elevation
        )
        
        # ========== 从球面坐标计算新的笛卡尔坐标 ==========
        # 球面坐标转笛卡尔坐标：
        # x = distance * cos(elevation) * cos(azimuth)
        # y = distance * cos(elevation) * sin(azimuth)
        # z = distance * sin(elevation)
        cos_elevation = np.cos(new_elevation)
        new_direction = np.array([
            distance * cos_elevation * np.cos(new_azimuth),
            distance * cos_elevation * np.sin(new_azimuth),
            distance * np.sin(new_elevation)
        ])
        
        # 计算新位置
        new_position = center + new_direction
        
        # ========== 更新摄像机 ==========
        camera.SetPosition(new_position)
        camera.SetFocalPoint(center)
        
        # ========== 更新view_up向量（保持相机不翻转）==========
        # 使用世界坐标的上向量（0,0,1）作为参考
        world_up = np.array([0.0, 0.0, 1.0])
        new_direction_normalized = new_direction / distance
        
        # 计算右向量（相机坐标系的右方向）
        right = np.cross(new_direction_normalized, world_up)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # 如果相机方向与world_up平行（几乎垂直），使用默认右向量
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        
        # 计算新的上向量（垂直于视线方向，尽量接近world_up）
        new_view_up = np.cross(right, new_direction_normalized)
        new_view_up = new_view_up / np.linalg.norm(new_view_up)
        
        camera.SetViewUp(new_view_up)
        
        view.render()
    
    @staticmethod
    def handle_pan(view, delta: QPoint):
        """处理平移操作"""
        camera = view.renderer.GetActiveCamera()
        
        # 获取当前摄像机参数
        center = np.array(camera.GetFocalPoint())
        position = np.array(camera.GetPosition())
        view_up = np.array(camera.GetViewUp())
        
        # 计算摄像机坐标系
        direction = position - center
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return
        forward = -direction / distance  # 指向中心的方向
        
        # 计算右向量和上向量
        right = np.cross(forward, view_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 计算平移距离（根据摄像机距离和窗口大小）
        window_size = view.size()
        min_window_size = max(min(window_size.width(), window_size.height()), 1)
        pan_sensitivity = distance / min_window_size * 0.5
        
        # 计算平移向量
        pan_x = -delta.x() * pan_sensitivity
        pan_y = delta.y() * pan_sensitivity
        
        # 应用平移
        new_center = center + right * pan_x + up * pan_y
        new_position = position + right * pan_x + up * pan_y
        
        # 更新摄像机
        camera.SetFocalPoint(new_center)
        camera.SetPosition(new_position)
        
        # 更新轨道中心
        view._orbit_center = new_center
        
        view.render()
    
    @staticmethod
    def handle_zoom_wheel(view, zoom_factor: float):
        """处理滚轮缩放"""
        camera = view.renderer.GetActiveCamera()
        
        center = np.array(camera.GetFocalPoint())
        position = np.array(camera.GetPosition())
        
        # 计算当前距离
        direction = position - center
        distance = np.linalg.norm(direction)
        
        # 计算初始距离（基于当前工作空间）
        initial_distance = view._calculate_initial_distance()
        
        # 应用缩放
        new_distance = distance / zoom_factor
        
        # 仅限制最大距离，不再限制最小距离
        max_distance = initial_distance * 5.0
        new_distance = min(new_distance, max_distance)
        
        # 更新摄像机位置
        direction_normalized = direction / distance
        new_position = center + direction_normalized * new_distance
        
        camera.SetPosition(new_position)
        view._camera_distance = new_distance
        
        view.render()
    
    @staticmethod
    def handle_zoom_drag(view, delta: QPoint):
        """处理拖拽缩放（Alt + 右键）"""
        # 垂直移动控制缩放
        zoom_sensitivity = 0.01
        zoom_factor = 1.0 - delta.y() * zoom_sensitivity
        
        CameraController.handle_zoom_wheel(view, zoom_factor)
    
    @staticmethod
    def get_camera_info(view) -> dict:
        """获取当前摄像机信息"""
        camera = view.renderer.GetActiveCamera()
        return {
            'position': np.array(camera.GetPosition()),
            'focal_point': np.array(camera.GetFocalPoint()),
            'view_up': np.array(camera.GetViewUp()),
            'distance': view._camera_distance,
            'orbit_center': view._orbit_center.copy()
        }
    
    @staticmethod
    def set_camera_info(view, camera_info: dict):
        """设置摄像机信息"""
        camera = view.renderer.GetActiveCamera()
        camera.SetPosition(camera_info['position'])
        camera.SetFocalPoint(camera_info['focal_point'])
        camera.SetViewUp(camera_info['view_up'])
        view._camera_distance = camera_info.get('distance', view._camera_distance)
        view._orbit_center = camera_info.get('orbit_center', view._orbit_center)
        view.render()
        view.view_changed.emit()
    
    @staticmethod
    def set_view(view, view_name: str):
        """设置快速视角"""
        camera = view.renderer.GetActiveCamera()
        center = view._orbit_center
        distance = view._camera_distance
        
        # 定义各个视角的方向向量和上向量
        views = {
            'front': {
                'direction': np.array([0.0, 1.0, 0.0]),  # 从-Y看向+Y
                'view_up': np.array([0.0, 0.0, 1.0])     # Z轴向上
            },
            'back': {
                'direction': np.array([0.0, -1.0, 0.0]),  # 从+Y看向-Y
                'view_up': np.array([0.0, 0.0, 1.0])
            },
            'top': {
                'direction': np.array([0.0, 0.0, 1.0]),   # 从-Z看向+Z（俯视）
                'view_up': np.array([0.0, 1.0, 0.0])      # Y轴向上（北向）
            },
            'bottom': {
                'direction': np.array([0.0, 0.0, -1.0]),  # 从+Z看向-Z（仰视）
                'view_up': np.array([0.0, 1.0, 0.0])
            },
            'left': {
                'direction': np.array([-1.0, 0.0, 0.0]),  # 从+X看向-X
                'view_up': np.array([0.0, 0.0, 1.0])
            },
            'right': {
                'direction': np.array([1.0, 0.0, 0.0]),   # 从-X看向+X
                'view_up': np.array([0.0, 0.0, 1.0])
            },
            'iso': {
                'direction': np.array([1.0, 1.0, 0.5]),   # 等轴测视图
                'view_up': np.array([0.0, 0.0, 1.0])
            }
        }
        
        if view_name not in views:
            view_name = 'iso'  # 默认使用等轴测视图
        
        view_config = views[view_name]
        direction = view_config['direction']
        direction = direction / np.linalg.norm(direction)
        
        # 计算摄像机位置
        camera_pos = center - direction * distance
        
        # 设置摄像机
        camera.SetPosition(camera_pos)
        camera.SetFocalPoint(center)
        camera.SetViewUp(view_config['view_up'])
        
        # 更新轨道中心
        view._orbit_center = center
        
        view.render()
        view.view_changed.emit()

    @staticmethod
    def focus_on_point(view, target_point: np.ndarray, zoom_factor: float = 0.5):
        """
        将视角聚焦到指定点
        """
        camera = view.renderer.GetActiveCamera()
        
        # 获取当前摄像机参数
        current_position = np.array(camera.GetPosition())
        current_focal = np.array(camera.GetFocalPoint())
        current_view_up = np.array(camera.GetViewUp())
        
        # 计算当前方向
        current_direction = current_position - current_focal
        current_distance = np.linalg.norm(current_direction)
        
        if current_distance < 1e-6:
            return  # 避免除零错误
        
        current_direction_normalized = current_direction / current_distance
        
        # 依据当前距离缩放，防止“移动不明显”
        bounds = view.workspace_bounds
        workspace_size = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        base_distance = current_distance * zoom_factor
        min_distance = max(workspace_size * 0.02, 1.0)
        max_distance = workspace_size * 2.0
        new_distance = np.clip(base_distance, min_distance, max_distance)
        
        # 计算新的摄像机位置（保持当前方向）
        new_position = target_point + current_direction_normalized * new_distance
        
        # 更新摄像机
        camera.SetPosition(new_position)
        camera.SetFocalPoint(target_point)
        camera.SetViewUp(current_view_up)
        
        # 更新轨道中心
        view._orbit_center = target_point.copy()
        view._camera_distance = new_distance
        
        view.render()
        view.view_changed.emit()
    
    @staticmethod
    def focus_on_plane(view, surface, distance_factor: float = 1.5):
        """将视角聚焦到面，摄像机沿着法线方向放置"""
        # 从面对象获取顶点和法向量
        plane_vertices = surface.vertices
        plane_normal = surface.normal
        
        # 计算面的中心点
        center = np.mean(plane_vertices, axis=0)
        
        # 第一步：使用 focus_on_point 聚焦到中心点（保持当前方向）
        CameraController.focus_on_point(view, center, zoom_factor=0.8)
        
        # 第二步：调整摄像机方向到法向量方向
        camera = view.renderer.GetActiveCamera()
        
        # 计算面的包围盒对角线长度
        bbox_min = np.min(plane_vertices, axis=0)
        bbox_max = np.max(plane_vertices, axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        
        # 计算目标距离
        target_distance = max(diag * distance_factor, 3.0)
        
        # 使用面提供的法向量
        normal_length = np.linalg.norm(plane_normal)
        if normal_length < 1e-10:
            # 法向量无效，使用默认方向
            plane_normal = np.array([0.0, 0.0, 1.0])
        else:
            plane_normal = plane_normal / normal_length
        
        # 计算新的摄像机位置：沿着法线方向放置
        camera_position = center + plane_normal * target_distance
        
        # 设置视角上向量：尽量保持Z轴向上
        # 如果法向量接近垂直，使用Y轴作为上向量
        if abs(plane_normal[2]) > 0.9:  # 法向量接近垂直
            view_up = np.array([0.0, 1.0, 0.0])
        else:
            view_up = np.array([0.0, 0.0, 1.0])
        
        # 更新摄像机（调整方向到法向量方向）
        camera.SetPosition(camera_position)
        camera.SetFocalPoint(center)
        camera.SetViewUp(view_up)
        
        # 更新轨道中心和距离
        view._orbit_center = center.copy()
        view._camera_distance = target_distance
        
        view.render()
        view.view_changed.emit()
    
    @staticmethod
    def focus_on_plane_by_coordinate(view, axis: str, coordinate: float, distance_factor: float = 1.5):
        """通过坐标直接聚焦到特定坐标的面"""
        # 获取工作空间边界
        workspace_bounds = view.workspace_bounds
        
        if axis.lower() == 'x':
            # 聚焦到x=coordinate的面
            center = np.array([coordinate, (workspace_bounds[2] + workspace_bounds[3]) / 2, (workspace_bounds[4] + workspace_bounds[5]) / 2])
            normal = np.array([1.0, 0.0, 0.0])
            # 计算面的范围
            y_range = workspace_bounds[3] - workspace_bounds[2]
            z_range = workspace_bounds[5] - workspace_bounds[4]
            diag = np.sqrt(y_range**2 + z_range**2)
            
        elif axis.lower() == 'y':
            # 聚焦到y=coordinate的面
            center = np.array([(workspace_bounds[0] + workspace_bounds[1]) / 2, coordinate, (workspace_bounds[4] + workspace_bounds[5]) / 2])
            normal = np.array([0.0, 1.0, 0.0])
            # 计算面的范围
            x_range = workspace_bounds[1] - workspace_bounds[0]
            z_range = workspace_bounds[5] - workspace_bounds[4]
            diag = np.sqrt(x_range**2 + z_range**2)
            
        elif axis.lower() == 'z':
            # 聚焦到z=coordinate的面
            center = np.array([(workspace_bounds[0] + workspace_bounds[1]) / 2, (workspace_bounds[2] + workspace_bounds[3]) / 2, coordinate])
            normal = np.array([0.0, 0.0, 1.0])
            # 计算面的范围
            x_range = workspace_bounds[1] - workspace_bounds[0]
            y_range = workspace_bounds[3] - workspace_bounds[2]
            diag = np.sqrt(x_range**2 + y_range**2)
            
        else:
            raise ValueError(f"无效的坐标轴: {axis}. 必须是 'x', 'y', 或 'z'")
        
        # 计算目标距离
        target_distance = max(diag * distance_factor, 5.0)
        
        # 获取当前摄像机
        camera = view.renderer.GetActiveCamera()
        
        # 计算新的摄像机位置：沿着法线方向放置
        camera_position = center + normal * target_distance
        
        # 设置视角上向量：尽量保持Z轴向上
        # 如果法向量接近垂直，使用Y轴作为上向量
        if abs(normal[2]) > 0.9:  # 法向量接近垂直
            view_up = np.array([0.0, 1.0, 0.0])
        else:
            view_up = np.array([0.0, 0.0, 1.0])
        
        # 更新摄像机
        camera.SetPosition(camera_position)
        camera.SetFocalPoint(center)
        camera.SetViewUp(view_up)
        
        # 更新轨道中心和距离
        view._orbit_center = center.copy()
        view._camera_distance = target_distance
        
        view.render()
        view.view_changed.emit()
        
        # 发送状态消息
        if hasattr(view, 'status_message'):
            view.status_message.emit(f'聚焦到 {axis.upper()}={coordinate:.2f} 的面')
    

