"""
坐标轴刻度组件
在3D视图中显示三个轴的刻度尺，带有刻度线和数字标注
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
import pyvista as pv


class AxisScaleComponent:
    """坐标轴刻度组件管理器"""
    
    def __init__(self, plotter):
        self.plotter = plotter
        self.scale_actors = {}  # 存储刻度尺的actor
        self.visible = False
        
    def create_axis_scales(self, bounds):
        """创建三个轴的刻度尺"""
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        # 清除旧的刻度尺
        self.clear_scales()
        
        # 1. X轴刻度尺 - 在y=0, z=0位置，红色大刻度
        self._create_axis_scale('x', x_min, x_max, y_min, z_min, 
                              y_pos=y_min, z_pos=z_min, 
                              large_color='red', axis_direction='x')
        
        # 2. Y轴刻度尺 - 在x=x_max, z=0位置，蓝色大刻度
        self._create_axis_scale('y', y_min, y_max, x_max, z_min,
                              x_pos=x_max, z_pos=z_min,
                              large_color='blue', axis_direction='y')
        
        # 3. Z轴刻度尺 - 在x=x_max, y=y_max位置，绿色大刻度
        self._create_axis_scale('z', z_min, z_max, x_max, y_max,
                              x_pos=x_max, y_pos=y_max,
                              large_color='green', axis_direction='z')
    
    def _create_axis_scale(self, axis_name, min_val, max_val, pos1, pos2, 
                          x_pos=None, y_pos=None, z_pos=None, 
                          large_color='red', axis_direction='x'):
        """创建单个轴的刻度尺（通用方法）"""
        # 计算刻度间隔：最大值的1/100为小刻度，每10个小刻度为一个大刻度
        value_range = max_val - min_val
        small_tick_interval = max(max_val, abs(min_val)) / 100  # 小刻度间隔
        large_tick_interval = small_tick_interval * 10      # 大刻度间隔
        
        # 计算刻度高度
        max_tick_height = self._calculate_max_tick_height(value_range, pos1, pos2)
        small_tick_height = max_tick_height * 0.5
        large_tick_height = max_tick_height
        
        # 创建主轴线
        main_actor = self._create_main_axis_line(axis_name, min_val, max_val, 
                                                x_pos, y_pos, z_pos, axis_direction)
        
        # 计算刻度数量
        num_small_ticks = int(value_range / small_tick_interval) + 1
        
        # 分别创建小刻度和大刻度的线
        small_tick_lines, large_tick_lines = self._generate_tick_lines(
            axis_name, min_val, max_val, small_tick_interval, num_small_ticks,
            small_tick_height, large_tick_height, x_pos, y_pos, z_pos, axis_direction
        )
        
        # 创建小刻度和大刻度mesh
        self._create_tick_meshes(axis_name, small_tick_lines, large_tick_lines, large_color)
        
        # 添加数字标注
        self._add_number_labels(axis_name, min_val, max_val, small_tick_interval,
                               num_small_ticks, large_tick_height, large_color,
                               x_pos, y_pos, z_pos, axis_direction)
    
    def _calculate_max_tick_height(self, value_range, pos1, pos2):
        """计算最大刻度高度 - 彻底切断位置与长度的关联
        
        核心原则：尺子的宽度不应该取决于它放在哪里
        无论尺子在原点还是1000米远，刻度长度都应该是固定的比例
        """
        # 完全忽略位置参数，只使用轴长度
        # 固定比例：大刻度长度 = 轴总长度的 6%（确保肉眼可见）
        # 最小值保底：即使轴很短也要有基本的刻度长度
        major_tick_length = max(value_range * 0.06, 2.0)  # 6%比例，最小2单位
        
        return major_tick_length
    
    def _create_main_axis_line(self, axis_name, min_val, max_val, x_pos, y_pos, z_pos, axis_direction):
        """创建主轴线"""
        if axis_direction == 'x':
            main_line_points = np.array([[min_val, y_pos, z_pos], [max_val, y_pos, z_pos]])
        elif axis_direction == 'y':
            main_line_points = np.array([[x_pos, min_val, z_pos], [x_pos, max_val, z_pos]])
        else:  # z
            main_line_points = np.array([[x_pos, y_pos, min_val], [x_pos, y_pos, max_val]])
        
        main_line = pv.PolyData(main_line_points)
        main_lines = np.array([2, 0, 1])
        main_line.lines = main_lines
        
        # 彻底回归线框模式 - 让PyVista自动推断为线条
        main_actor = self.plotter.add_mesh(
            main_line,
            color='darkgray',      # 更深的颜色，作为脊柱
            line_width=5,          # 最粗的线宽，作为尺子骨架
            style='wireframe',
            render_points_as_spheres=False,
            point_size=0,
            show_vertices=False,
            render_lines_as_tubes=False,
            pickable=False,
            name=f'{axis_name}_axis_main',
        )
        self.scale_actors[f'{axis_name}_axis_main'] = main_actor
        return main_actor
    
    def _generate_tick_lines(self, axis_name, min_val, max_val, small_tick_interval, 
                           num_small_ticks, small_tick_height, large_tick_height,
                           x_pos, y_pos, z_pos, axis_direction):
        """生成刻度线段"""
        small_tick_lines = []
        large_tick_lines = []
        
        for i in range(num_small_ticks + 1):
            value = min_val + i * small_tick_interval
            
            # 跳过超出范围的刻度
            if value > max_val:
                break
            
            # 判断是否为大刻度
            is_large_tick = (i % 10 == 0)
            tick_height = large_tick_height if is_large_tick else small_tick_height
            
            # 创建刻度线段 - 垂直于轴线
            if axis_direction == 'x':
                # X轴刻度：向Y轴负方向延伸
                line = [[value, y_pos, z_pos], [value, y_pos - tick_height, z_pos]]
            elif axis_direction == 'y':
                # Y轴刻度：向X轴正方向延伸
                line = [[x_pos, value, z_pos], [x_pos + tick_height, value, z_pos]]
            else:  # z
                # Z轴刻度：向X轴正方向延伸
                line = [[x_pos, y_pos, value], [x_pos + tick_height, y_pos, value]]
            
            if is_large_tick:
                large_tick_lines.append(line)
            else:
                small_tick_lines.append(line)
        
        return small_tick_lines, large_tick_lines
    
    def _create_tick_meshes(self, axis_name, small_tick_lines, large_tick_lines, large_color):
        """创建小刻度和大刻度mesh - 建立三级视觉层级
        
        视觉层级设计：
        1. 脊柱（主轴线）：5px，darkgray（最粗最深）
        2. 大刻度：3px，彩色（中等粗细，读数关键）
        3. 小刻度：1px，lightgray（最细，辅助）
        """
        # 创建小刻度mesh（最细，辅助线）
        if small_tick_lines:
            small_actor = self._create_lines_mesh(small_tick_lines, 'lightgray', 1, 
                                                 f'{axis_name}_small_ticks')
            self.scale_actors[f'{axis_name}_small_ticks'] = small_actor
        
        # 创建大刻度mesh（中等粗细，读数关键）
        if large_tick_lines:
            large_actor = self._create_lines_mesh(large_tick_lines, large_color, 3, 
                                                 f'{axis_name}_large_ticks')
            self.scale_actors[f'{axis_name}_large_ticks'] = large_actor
    
    def _create_lines_mesh(self, lines, color, line_width, name):
        """创建线段mesh的通用方法 - 优化以隐藏端点"""
        points = []
        line_indices = []
        point_count = 0
        
        for line in lines:
            points.extend(line)
            line_indices.extend([2, point_count, point_count + 1])
            point_count += 2
        
        points = np.array(points)
        mesh = pv.PolyData(points)
        mesh.lines = np.array(line_indices)

        # 强制仅渲染线单元，关闭一切点可视化
        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            line_width=line_width,
            style='wireframe',
            render_points_as_spheres=False,
            point_size=0,
            show_vertices=False,
            render_lines_as_tubes=False,
            pickable=False,
            name=name,
        )
        return actor
    
    def _add_number_labels(self, axis_name, min_val, max_val, small_tick_interval,
                          num_small_ticks, large_tick_height, large_color,
                          x_pos, y_pos, z_pos, axis_direction):
        """添加数字标注"""
        for i in range(0, num_small_ticks + 1, 10):  # 每10个刻度
            value = min_val + i * small_tick_interval
            if value <= max_val:
                # 优化文字位置 - 动态跟随刻度长度
                # 文字偏移量 = 刻度长度 * 1.2 + 固定值，确保不会压在刻度线上
                text_offset = large_tick_height * 1.2 + 2.0  # 2.0为固定偏移保底
                
                if axis_direction == 'x':
                    # X轴数字：向Y轴负方向偏移（与刻度线方向一致）
                    text_pos = [value, y_pos - text_offset, z_pos]
                elif axis_direction == 'y':
                    # Y轴数字：向X轴正方向偏移（与刻度线方向一致）
                    text_pos = [x_pos + text_offset, value, z_pos]
                else:  # z
                    # Z轴数字：向X轴正方向偏移（与刻度线方向一致）
                    text_pos = [x_pos + text_offset, y_pos, value]
                
                self._add_text_label(f'{value:.0f}', text_pos, large_color, 
                                   f'{axis_name}_label_{i//10}')
        
    def _add_text_label(self, text, position, color, name):
        """添加文字标注"""
        try:
            # 使用PyVista添加文字，去除背景框
            # 注意：render_points 和 opacity 参数在新版本 PyVista 中已移除
            text_actor = self.plotter.add_point_labels(
                [position],
                [text],
                point_size=0,
                font_size=12,
                text_color=color,
                name=name,
                shape=None,           # 去除背景框
                show_points=False,    # 确保不显示点
                always_visible=True   # 确保文字始终可见
            )
            self.scale_actors[name] = text_actor
        except Exception as e:
            print(f"添加文字标注失败: {e}")
            # 如果文字添加失败，只保留刻度线
    
    def _calculate_tick_interval(self, value_range):
        """计算合适的刻度间隔"""
        if value_range <= 0:
            return 10.0
            
        # 标准刻度间隔
        standard_intervals = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]
        
        # 理想的刻度数量（5-8个）
        ideal_ticks = 6
        ideal_interval = value_range / ideal_ticks
        
        # 找到最接近的标准间隔
        closest_interval = standard_intervals[0]
        for interval in standard_intervals:
            if abs(interval - ideal_interval) < abs(closest_interval - ideal_interval):
                closest_interval = interval
                
        return closest_interval
    
    def clear_scales(self):
        """清除所有刻度尺"""
        for name in list(self.scale_actors.keys()):
            try:
                actor = self.scale_actors[name]
                self.plotter.remove_actor(actor)
            except:
                pass
        self.scale_actors.clear()
    
    def set_visible(self, visible):
        """设置刻度尺可见性"""
        self.visible = visible
        
        if visible:
            # 重新创建刻度尺
            if hasattr(self.plotter, 'workspace_bounds'):
                self.create_axis_scales(self.plotter.workspace_bounds)
        else:
            # 清除刻度尺
            self.clear_scales()
        
        # 渲染更新
        self.plotter.render()
    
    def get_visible(self):
        """获取刻度尺可见性"""
        return self.visible
    
    def toggle_visible(self):
        """切换刻度尺可见性"""
        self.set_visible(not self.visible)
        return self.visible
    
    def update_bounds(self, bounds):
        """更新工作空间边界"""
        if self.visible:
            self.create_axis_scales(bounds)
