"""
方向组件 - 固定在屏幕右上角，显示当前视角方向
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygon
import numpy as np
from typing import Optional


class ViewAxes2D(QWidget):
    """方向组件 - 显示当前视角方向的坐标轴"""
    
    def __init__(self, parent=None, size=100):
        """
        初始化方向组件
        
        Parameters:
        -----------
        parent : QWidget, optional
            父窗口
        size : int
            控件大小（像素），默认100
        """
        super().__init__(parent)
        
        self._size = size
        self.setFixedSize(size, size)
        
        # 设置背景为透明
        self.setStyleSheet("background-color: transparent;")
        
        # 相机方向（归一化向量）
        self._camera_direction = np.array([0.0, 1.0, 0.0])  # 默认看向+Y
        self._camera_up = np.array([0.0, 0.0, 1.0])  # 默认Z轴向上
        
        # 坐标轴颜色（浅灰色系，与白色背景协调）
        self._x_color = QColor(180, 100, 100)  # 浅红灰色 - X轴
        self._y_color = QColor(100, 180, 100)  # 浅绿灰色 - Y轴
        self._z_color = QColor(100, 100, 180)  # 浅蓝灰色 - Z轴
    
    def update_camera_direction(self, camera_direction: np.ndarray, camera_up: np.ndarray):
        """
        更新相机方向
        
        Parameters:
        -----------
        camera_direction : np.ndarray
            相机方向向量（从焦点指向相机）
        camera_up : np.ndarray
            相机上向量
        """
        # 归一化向量
        direction_norm = np.linalg.norm(camera_direction)
        up_norm = np.linalg.norm(camera_up)
        
        if direction_norm > 1e-6:
            self._camera_direction = camera_direction / direction_norm
        if up_norm > 1e-6:
            self._camera_up = camera_up / up_norm
        
        # 触发重绘（只重绘一次，避免重影）
        self.update()
    
    def paintEvent(self, event):
        """绘制坐标轴"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 先清除整个区域，使用白色背景（避免重影）
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # 使用标准合成模式
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        
        # 获取控件大小
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        
        # 计算坐标轴长度（留出边距）
        margin = 15
        axis_length = min(width, height) / 2 - margin
        
        # 计算坐标轴在世界坐标系中的方向
        # 默认：X轴=[1,0,0], Y轴=[0,1,0], Z轴=[0,0,1]
        world_x = np.array([1.0, 0.0, 0.0])
        world_y = np.array([0.0, 1.0, 0.0])
        world_z = np.array([0.0, 0.0, 1.0])
        
        # 计算相机坐标系
        # 相机看向的方向（从相机指向焦点，与camera_direction相反）
        view_direction = -self._camera_direction
        
        # 计算右向量和上向量
        right = np.cross(view_direction, self._camera_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # 将世界坐标系的轴投影到屏幕坐标系
        # 屏幕坐标系：X向右，Y向下（需要翻转Y）
        def project_to_screen(world_vec):
            """将世界坐标向量投影到屏幕坐标"""
            # 计算在相机坐标系中的分量
            # right: 相机右向量（屏幕X方向）
            # up: 相机上向量（屏幕Y方向，需要翻转）
            # view_direction: 相机看向的方向（屏幕Z方向，深度）
            x_component = np.dot(world_vec, right)
            y_component = np.dot(world_vec, up)
            z_component = np.dot(world_vec, view_direction)
            
            # 使用正交投影：忽略Z分量（深度），只显示XY投影
            # 这样坐标轴会始终显示在2D平面上
            screen_x = x_component * axis_length
            screen_y = -y_component * axis_length  # 翻转Y轴（屏幕Y向下）
            
            return screen_x, screen_y
        
        # 绘制三个轴
        axes = [
            (world_x, self._x_color, 'X'),
            (world_y, self._y_color, 'Y'),
            (world_z, self._z_color, 'Z')
        ]
        
        for world_vec, color, label in axes:
            # 计算屏幕坐标
            screen_x, screen_y = project_to_screen(world_vec)
            end_x = center_x + screen_x
            end_y = center_y + screen_y
            
            # 绘制轴线（使用圆角端点，避免重影）
            pen = QPen(color, 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(int(center_x), int(center_y), int(end_x), int(end_y))
            
            # 绘制箭头
            arrow_size = 8
            # 计算箭头方向
            arrow_dir = np.array([screen_x, screen_y])
            arrow_dir = arrow_dir / np.linalg.norm(arrow_dir) if np.linalg.norm(arrow_dir) > 1e-6 else np.array([1, 0])
            
            # 箭头顶点
            arrow_tip = np.array([end_x, end_y])
            arrow_perp = np.array([-arrow_dir[1], arrow_dir[0]])  # 垂直向量
            
            arrow_point1 = arrow_tip - arrow_dir * arrow_size + arrow_perp * arrow_size * 0.5
            arrow_point2 = arrow_tip - arrow_dir * arrow_size - arrow_perp * arrow_size * 0.5
            
            arrow_polygon = QPolygon([
                QPoint(int(end_x), int(end_y)),
                QPoint(int(arrow_point1[0]), int(arrow_point1[1])),
                QPoint(int(arrow_point2[0]), int(arrow_point2[1]))
            ])
            
            painter.setBrush(QBrush(color))
            painter.drawPolygon(arrow_polygon)
            
            # 绘制标签
            label_offset = 12
            label_x = end_x + arrow_dir[0] * label_offset
            label_y = end_y + arrow_dir[1] * label_offset
            
            painter.setPen(QPen(color, 1))
            font = QFont('Arial', 10, QFont.Bold)
            painter.setFont(font)
            painter.drawText(int(label_x - 5), int(label_y + 5), label)
        
        # 绘制中心点（浅灰色）
        center_color = QColor(150, 150, 150)
        painter.setPen(QPen(center_color, 1))
        painter.setBrush(QBrush(center_color))
        painter.drawEllipse(int(center_x - 3), int(center_y - 3), 6, 6)

