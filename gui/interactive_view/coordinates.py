"""
坐标转换相关方法
"""
from PyQt6.QtCore import QPoint
import numpy as np
from typing import Optional


class CoordinateConverter:
    """坐标转换器 - 用于屏幕坐标到世界坐标的转换"""

    @staticmethod
    def screen_to_horizontal_plane(
        view,
        screen_pos: QPoint,
        z_value: float,
        clip_to_bounds: bool = True,
    ) -> Optional[np.ndarray]:
        """将屏幕坐标投影到指定 Z 高程的水平平面。"""
        return CoordinateConverter.screen_to_axis_aligned_plane(
            view,
            screen_pos,
            axis="z",
            axis_value=z_value,
            clip_to_bounds=clip_to_bounds,
        )

    @staticmethod
    def screen_to_axis_aligned_plane(
        view,
        screen_pos: QPoint,
        *,
        axis: str,
        axis_value: float,
        clip_to_bounds: bool = True,
    ) -> Optional[np.ndarray]:
        """将屏幕坐标投影到轴对齐平面（X/Y/Z 其中一轴固定）。"""
        try:
            axis_name = str(axis).strip().lower()
            axis_index_map = {"x": 0, "y": 1, "z": 2}
            if axis_name not in axis_index_map:
                return None
            axis_index = axis_index_map[axis_name]

            renderer = view.renderer
            height = max(view.height(), 1)
            vtk_x = screen_pos.x()
            vtk_y = height - screen_pos.y() - 1

            renderer.SetDisplayPoint(vtk_x, vtk_y, 0.0)
            renderer.DisplayToWorld()
            near_world = renderer.GetWorldPoint()
            renderer.SetDisplayPoint(vtk_x, vtk_y, 1.0)
            renderer.DisplayToWorld()
            far_world = renderer.GetWorldPoint()

            if abs(near_world[3]) < 1e-12 or abs(far_world[3]) < 1e-12:
                return None

            near_point = np.array(near_world[:3], dtype=float) / near_world[3]
            far_point = np.array(far_world[:3], dtype=float) / far_world[3]
            ray = far_point - near_point
            ray_component = float(ray[axis_index])
            if abs(ray_component) < 1e-12:
                return None

            factor = (float(axis_value) - float(near_point[axis_index])) / ray_component
            world_pos = near_point + ray * factor

            if clip_to_bounds:
                world_pos[0] = np.clip(world_pos[0], view.workspace_bounds[0], view.workspace_bounds[1])
                world_pos[1] = np.clip(world_pos[1], view.workspace_bounds[2], view.workspace_bounds[3])
                world_pos[2] = np.clip(world_pos[2], view.workspace_bounds[4], view.workspace_bounds[5])
            return world_pos
        except Exception:
            return None
    
    @staticmethod
    def screen_to_world_raycast(view, screen_pos: QPoint) -> Optional[np.ndarray]:
        """使用射线投射获取鼠标指向的世界坐标（与场景的交点）"""
        try:
            # 使用PyVista的pick功能进行射线投射
            try:
                picked = view.pick_mouse_position()
                if picked and hasattr(picked, 'point'):
                    return np.array(picked.point)
            except:
                pass
            
            # 如果pick_mouse_position不可用，尝试使用VTK的WorldPointPicker
            try:
                from vtkmodules.vtkRenderingCore import vtkWorldPointPicker
                width = view.width()
                height = view.height()
                vtk_x = screen_pos.x()
                vtk_y = height - screen_pos.y() - 1
                world_picker = vtkWorldPointPicker()
                world_picker.Pick(vtk_x, vtk_y, 0, view.renderer)
                picked_pos = world_picker.GetPickPosition()
                if picked_pos and any(abs(p) > 1e-6 for p in picked_pos):
                    return np.array(picked_pos)
            except:
                pass
        except Exception:
            pass
    
    @staticmethod
    def screen_to_world(view, screen_pos: QPoint, depth: float = 0.0, clip_to_bounds: bool = True) -> Optional[np.ndarray]:
        """将屏幕坐标转换为世界坐标"""
        try:
            # 获取渲染器
            renderer = view.renderer
            
            # 获取屏幕尺寸
            width = view.width()
            height = view.height()
            
            # 将Qt坐标转换为VTK坐标（Y轴翻转）
            vtk_x = screen_pos.x()
            vtk_y = height - screen_pos.y() - 1
            
            # 使用VTK的屏幕到世界坐标转换
            # 首先获取焦点平面的点
            world_pos = [0.0, 0.0, 0.0]
            renderer.SetDisplayPoint(vtk_x, vtk_y, depth)
            renderer.DisplayToWorld()
            world_pos = renderer.GetWorldPoint()
            
            if world_pos[3] != 0.0:
                # 齐次坐标转换为3D坐标
                world_pos = np.array([
                    world_pos[0] / world_pos[3],
                    world_pos[1] / world_pos[3],
                    world_pos[2] / world_pos[3]
                ])
                
                # 如果启用限制，将坐标限制在工作空间内部（包含边界）
                if clip_to_bounds:
                    # 限制X坐标在空间内部（包含边界）
                    world_pos[0] = np.clip(world_pos[0], 
                                         view.workspace_bounds[0], 
                                         view.workspace_bounds[1])
                    # 限制Y坐标在空间内部（包含边界）
                    world_pos[1] = np.clip(world_pos[1], 
                                         view.workspace_bounds[2], 
                                         view.workspace_bounds[3])
                    # 限制Z坐标在空间内部（包含边界）
                    world_pos[2] = np.clip(world_pos[2], 
                                         view.workspace_bounds[4], 
                                         view.workspace_bounds[5])
                
                return world_pos
            else:
                return None
        except Exception as e:
            print(f"屏幕坐标转换失败: {e}")
            return None
    
    @staticmethod
    def screen_to_plane_relative(view, screen_pos: QPoint, plane_vertices: np.ndarray) -> Optional[np.ndarray]:
        """
        获取光标在选中平面中的相对位置（2D平面坐标）
        当视角移动到平面法线方向（正上方）后，将屏幕坐标投影到平面上，
        并返回在平面局部坐标系中的2D相对位置
        """
        try:
            if plane_vertices is None or len(plane_vertices) < 3:
                return None
            
            # 计算平面的原点、法线和局部坐标系
            p0 = plane_vertices[0]  # 平面原点
            v1 = plane_vertices[1] - p0  # 第一个方向向量
            v2 = plane_vertices[2] - p0  # 第二个方向向量
            
            # 计算平面法线
            normal = np.cross(v1, v2)
            normal_len = np.linalg.norm(normal)
            if normal_len < 1e-8:
                return None
            normal = normal / normal_len
            
            # 构建平面局部坐标系的基向量
            # U轴：沿着第一个边的方向
            u_axis = v1 / np.linalg.norm(v1)
            # V轴：垂直于U轴和法线
            v_axis = np.cross(normal, u_axis)
            v_axis = v_axis / np.linalg.norm(v_axis)
            
            # 从屏幕坐标获取射线
            renderer = view.renderer
            camera = renderer.GetActiveCamera()
            
            # 将屏幕坐标转换为VTK坐标
            width = view.width()
            height = view.height()
            vtk_x = screen_pos.x()
            vtk_y = height - screen_pos.y() - 1
            
            # 获取射线的起点和方向
            # 使用DisplayToWorld转换获取近平面和远平面的点
            renderer.SetDisplayPoint(vtk_x, vtk_y, 0.0)
            renderer.DisplayToWorld()
            near_point = np.array(renderer.GetWorldPoint()[:3]) / renderer.GetWorldPoint()[3]
            
            renderer.SetDisplayPoint(vtk_x, vtk_y, 1.0)
            renderer.DisplayToWorld()
            far_point = np.array(renderer.GetWorldPoint()[:3]) / renderer.GetWorldPoint()[3]
            
            # 射线方向
            ray_dir = far_point - near_point
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # 计算射线与平面的交点
            # 平面方程: dot(normal, P - p0) = 0
            # 射线方程: P = near_point + t * ray_dir
            # 求解 t: dot(normal, near_point + t * ray_dir - p0) = 0
            denom = np.dot(normal, ray_dir)
            if abs(denom) < 1e-8:
                # 射线与平面平行
                return None
            
            t = np.dot(normal, p0 - near_point) / denom
            if t < 0:
                # 交点在射线起点后面
                return None
            
            # 计算交点
            intersection = near_point + t * ray_dir
            
            # 将交点转换到平面局部坐标系
            vec = intersection - p0
            u = np.dot(vec, u_axis)
            v = np.dot(vec, v_axis)
            
            return np.array([u, v])
            
        except Exception as e:
            print(f"平面相对坐标转换失败: {e}")
            return None
    
    @staticmethod
    def screen_to_world_on_plane(view, screen_pos: QPoint, plane_vertices: np.ndarray) -> Optional[np.ndarray]:
        """
        将屏幕坐标通过选中平面转换为世界坐标
        先获取在平面上的相对位置，再转换为世界坐标
        """
        # 第一步：获取平面相对坐标
        relative_pos = CoordinateConverter.screen_to_plane_relative(view, screen_pos, plane_vertices)
        if relative_pos is None:
            return None
        
        # 第二步：转换为世界坐标
        return CoordinateConverter.plane_relative_to_world(plane_vertices, relative_pos)
    
    @staticmethod
    def plane_relative_to_world(plane_vertices: np.ndarray, relative_pos: np.ndarray) -> Optional[np.ndarray]:
        """将平面内的2D相对位置转换为3D世界坐标"""
        try:
            if plane_vertices is None or len(plane_vertices) < 3:
                return None
            
            if relative_pos is None or len(relative_pos) != 2:
                return None
            
            # 计算平面的原点和局部坐标系
            p0 = plane_vertices[0]  # 平面原点
            v1 = plane_vertices[1] - p0  # 第一个方向向量
            v2 = plane_vertices[2] - p0  # 第二个方向向量
            
            # 计算平面法线
            normal = np.cross(v1, v2)
            normal_len = np.linalg.norm(normal)
            if normal_len < 1e-8:
                return None
            normal = normal / normal_len
            
            # 构建平面局部坐标系的基向量
            # U轴：沿着第一个边的方向
            u_axis = v1 / np.linalg.norm(v1)
            # V轴：垂直于U轴和法线
            v_axis = np.cross(normal, u_axis)
            v_axis = v_axis / np.linalg.norm(v_axis)
            
            # 将局部坐标转换为世界坐标
            u, v = relative_pos[0], relative_pos[1]
            world_pos = p0 + u * u_axis + v * v_axis
            
            return world_pos
            
        except Exception as e:
            print(f"平面坐标转世界坐标失败: {e}")
            return None
   
