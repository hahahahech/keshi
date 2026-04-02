"""

场景管理面板
三维地质建模系统的场景管理器组件，用于管理剖面、曲面、块体等对象

"""

import re
import traceback
from typing import Optional, Dict, Any, List
# PyQt6 imports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QCheckBox, QLabel, QSpinBox, QHeaderView, QFrame, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap, QAction
# 不再使用SceneObjectMetadata，直接使用业务对象

class UserRole:

    """用户角色数据键"""

    METADATA = Qt.ItemDataRole.UserRole + 1
    OBJECT_TYPE = Qt.ItemDataRole.UserRole + 2

class ColorIcon(QLabel):

    """自定义颜色图标标签"""

    def __init__(self, color, size=16):

        super().__init__()
        self.color = QColor(color)
        self.size = size
        self.setFixedSize(size, size)
        self.update_pixmap()

    def update_pixmap(self):

        """更新颜色图标"""

        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 绘制带边框的色块
        painter.fillRect(1, 1, self.size-2, self.size-2, self.color)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawRect(0, 0, self.size-1, self.size-1)
        painter.end()
        self.setPixmap(pixmap)

    def set_color(self, color):

        """设置颜色"""

        self.color = QColor(color)
        self.update_pixmap()

class OpacitySpinBox(QSpinBox):

    """自定义透明度SpinBox"""

    def __init__(self):

        super().__init__()
        self.setRange(0, 100)
        self.setSingleStep(1)
        self.setSuffix("%")
        self.setMaximumWidth(60)
        # 设置紧凑样式

        self.setStyleSheet("""

            QSpinBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 2px 6px;
                background-color: white;
                font-size: 11px;
            }
            QSpinBox:focus {
                border: 1px solid #4a90e2;
                outline: none;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                border: none;
                background-color: #f5f5f5;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #e0e0e0;
            }

        """)

class SceneTreeWidgetItem(QTreeWidgetItem):

    """场景树项 - 双向绑定代理

    持有业务对象的引用，UI操作直接反映到业务对象，
    业务对象的变化也可以通过此节点刷新UI显示

    """

    def __init__(self, parent=None, data_object=None):

        super().__init__(parent)
        self.data_object = data_object # 业务对象实例（Section/Block/Surface/Line等）
        self.item_type = "object" # 兼容旧代码
        self.checkbox = None
        self.color_icon = None
        self.opacity_spinbox = None
        self.interaction_state = "normal" # normal, editing, selected
        # 如果提供了业务对象，自动初始化UI
        if data_object:
            self._init_from_data_object()

    @property

    def object_id(self):

        """从业务对象获取ID"""

        if self.data_object:
            return getattr(self.data_object, 'line_id', None) or \
                   getattr(self.data_object, 'model_id', None) or \
                   getattr(self.data_object, 'surface_id', None) or \
                   getattr(self.data_object, 'section_id', None) or \
                   getattr(self.data_object, 'block_id', None) or \
                   getattr(self.data_object, 'id', None)
        return getattr(self, '_object_id', None)

    @object_id.setter

    def object_id(self, value):

        """兼容旧代码的setter"""

        self._object_id = value

    def _init_from_data_object(self):

        """从业务对象初始化UI显示"""

        if not self.data_object:
            return

        obj = self.data_object
        # 获取对象属性
        name = getattr(obj, 'name', str(self.object_id))
        visible = getattr(obj, 'visible', True)
        color = getattr(obj, 'color', (0.5, 0.5, 0.5))
        opacity = getattr(obj, 'opacity', 1.0) if hasattr(obj, 'opacity') else getattr(obj, 'transparency', 0.8)
        # 转换颜色格式
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        else:
            color_hex = "#808080"
        # 设置UI显示
        self.setCheckState(0, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self.setText(0, f" {name[:20]}")
        self.setToolTip(0, name)
        # 设置颜色图标
        self.color_icon = ColorIcon(color_hex, 14)
        self.setIcon(0, QIcon(self.color_icon.pixmap()))
        # 设置ID显示
        obj_id = self.object_id
        display_id = str(obj_id)
        match = re.search(r'(\d+)', str(obj_id))
        if match:
            display_id = match.group(1)
        self.setText(1, display_id)
        # 设置透明度
        opacity_percent = int(opacity * 100)
        self.opacity_spinbox = OpacitySpinBox()
        self.opacity_spinbox.setValue(opacity_percent)
        # 连接透明度变化信号
        self.opacity_spinbox.valueChanged.connect(self._on_opacity_changed)
        tree = self.treeWidget()
        if tree:
            tree.setItemWidget(self, 2, self.opacity_spinbox)

    def refresh_display(self):

        """刷新UI显示（当业务对象变化时调用）"""

        if self.data_object:
            self._init_from_data_object()

    def _on_opacity_changed(self, value):

        """透明度变化处理"""

        try:
            if self.data_object:
                # 获取场景管理器
                tree = self.treeWidget()
                if tree and hasattr(tree.parent(), 'opacityChanged'):
                    # 发出透明度变化信号
                    tree.parent().opacityChanged.emit(self.object_id, value)
        except Exception as e:
            print(f"透明度变化处理失败: {e}")

    def set_interaction_state(self, state):

        """设置交互状态

        Parameters:
        -----------
        state : str
            状态类型: "normal", "editing", "selected"

        """

        self.interaction_state = state
        # 根据状态更新视觉样式
        if state == "editing":
            # 编辑状态：红色背景
            self.setBackground(0, QColor(255, 200, 200))
            self.setBackground(1, QColor(255, 200, 200))
            self.setBackground(2, QColor(255, 200, 200))
        elif state == "selected":
            # 选中状态：蓝色背景
            self.setBackground(0, QColor(200, 220, 255))
            self.setBackground(1, QColor(200, 220, 255))
            self.setBackground(2, QColor(200, 220, 255))
        else:
            # 正常状态：默认背景
            self.setBackground(0, QColor())
            self.setBackground(1, QColor())
            self.setBackground(2, QColor())

    def toggle_wireframe(self):

        """切换线框显示模式"""

        # 这个方法可以与 _apply_to_actors 配合使用
        pass

class SceneManagerPanel(QWidget):

    """场景管理面板"""

    # 自定义信号
    visibilityChanged = pyqtSignal(str, bool) # id, visible
    opacityChanged = pyqtSignal(str, int) # id, opacity
    objectSelected = pyqtSignal(str) # id - 对象被选中
    objectDeselected = pyqtSignal(str) # id - 对象取消选中

    def __init__(self, parent=None):

        super().__init__(parent)
        # 保存parent引用
        self.parent_window = parent
        # 根节点字典
        self.root_nodes = {}
        # 3D视图引用
        self.plotter = None
        # 批量更新标志（静默模式）
        self._batch_update_mode = False
        self._batch_update_depth = 0
        self.setup_ui()
        self.setup_connections()

    def begin_batch_update(self):

        """开始批量更新（进入静默模式）

        在批量添加节点时调用，阻塞信号避免重复触发

        """

        self._batch_update_depth += 1
        if self._batch_update_depth == 1:
            self._batch_update_mode = True
            self.tree_widget.blockSignals(True)

    def end_batch_update(self):

        """结束批量更新（退出静默模式）

        批量添加完成后调用，恢复信号并刷新显示

        """

        self._batch_update_depth = max(0, self._batch_update_depth - 1)
        if self._batch_update_depth == 0:
            self._batch_update_mode = False
            self.tree_widget.blockSignals(False)
            # 统一刷新渲染
            if self.plotter:
                self.plotter.render()

    def setup_ui(self):

        """设置界面"""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)
        # 创建树形控件
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["场景节点", "ID", "透明度"])
        self.tree_widget.setRootIsDecorated(True) # 确保根节点显示展开/收起图标
        # 启用右键菜单
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        # 设置表头样式
        header = self.tree_widget.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        # 设置树形控件样式
        self.tree_widget.setStyleSheet("")
        # 创建根节点
        self.create_root_nodes()
        layout.addWidget(self.tree_widget)

    def create_root_nodes(self):

        """创建根节点"""

        categories = ["模型", "剖面", "曲面", "块体"]
        for cat in categories:
            node = QTreeWidgetItem(self.tree_widget)
            node.setText(0, cat)
            node.setExpanded(True)
            # 核心：必须允许展开
            node.setFlags(node.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.root_nodes[cat] = node

    def setup_connections(self):

        """设置信号连接"""

        # 这里需要通过事件过滤器来处理子控件信号
        self.tree_widget.itemChanged.connect(self.on_item_changed)
        # 连接选中信号
        self.tree_widget.itemSelectionChanged.connect(self.on_selection_changed)
        # 连接自己的透明度变化信号到处理方法
        self.opacityChanged.connect(self.on_scene_opacity_changed)

    def _get_object_type_from_id(self, obj_id, item_type="object"):

        """根据ID和类型确定对象类型字符串"""

        if item_type == "line":
            return "line"

        elif obj_id.startswith('pem_'):
            return "section"

        elif obj_id.startswith('model_'):
            return "model"

        elif obj_id.startswith('surface_'):
            return "surface"

        elif obj_id.startswith('block_'):
            return "block"

        else:
            return "object"

    def _get_object_type(self, item):

        """获取项目的对象类型字符串"""

        # 优先从data_object获取类型
        if hasattr(item, 'data_object') and item.data_object:
            if hasattr(item.data_object, 'line_id'):
                return "line"

            elif hasattr(item.data_object, 'model_id'):
                return "model"

            elif hasattr(item.data_object, 'surface_id'):
                return "surface"

            elif hasattr(item.data_object, 'section_id'):
                return "section"

            elif hasattr(item.data_object, 'block_id'):
                return "block"

        # 回退到ID判断
        if item.object_id:
            return self._get_object_type_from_id(item.object_id, getattr(item, 'item_type', 'object'))

        return "object"

    def add_object(self, data_object, category=None):

        """通用对象添加方法 - 新架构核心方法（增强版）

        Parameters:
        -----------
        data_object : object
            业务对象实例（Line/Surface/Section/Block等）
        category : str, optional
            类别名称，如果不提供则自动推断
        Returns:
        --------
        SceneTreeWidgetItem : 创建的树节点

        """

        # 自动推断类别
        if category is None:
            if hasattr(data_object, 'line_id'):
                category = '剖面' # 线条属于剖面
            elif hasattr(data_object, 'model_id'):
                category = '模型'
            elif hasattr(data_object, 'surface_id'):
                category = '曲面'
            elif hasattr(data_object, 'section_id'):
                category = '剖面'
            elif hasattr(data_object, 'block_id'):
                category = '块体'
            else:
                raise ValueError(f"无法推断对象类别: {type(data_object)}")
        # 获取根类别节点
        root_parent_node = self.root_nodes.get(category)
        if not root_parent_node:
            raise ValueError(f"未知的类别: {category}")
        # 智能父子关系处理
        parent_node = self._find_parent_node_intelligently(data_object, root_parent_node)
        # 创建节点（使用新的双向绑定方式）
        item = SceneTreeWidgetItem(parent=parent_node, data_object=data_object)
        # 确保父节点展开
        parent_node.setExpanded(True)
        # 如果挂载到根类别节点，确保根节点也展开
        if parent_node in self.root_nodes.values():
            parent_node.setExpanded(True)
        # 添加对象到场景树
        return item

    def get_object_by_id(self, obj_id):
        """根据ID获取业务对象"""
        item = self._find_item_by_id(obj_id)
        if item and hasattr(item, 'data_object'):
            return item.data_object
        return None

    def _find_item_by_id(self, obj_id):

        """在树中查找指定ID的节点"""

        for root in self.root_nodes.values():
            item = self._find_item_recursive(root, obj_id)
            if item:
                return item

        return None

    def _find_item_recursive(self, parent, obj_id):

        """递归查找节点"""

        for i in range(parent.childCount()):
            child = parent.child(i)
            if isinstance(child, SceneTreeWidgetItem) and child.object_id == obj_id:
                return child

            # 递归查找子节点
            result = self._find_item_recursive(child, obj_id)
            if result:
                return result

        return None

    def _find_parent_node_intelligently(self, data_object, root_parent_node):

        """智能查找父节点

        Parameters:
        -----------
        data_object : object
            业务对象
        root_parent_node : QTreeWidgetItem
            根类别节点（作为兜底）
        Returns:
        --------
        QTreeWidgetItem : 找到的父节点

        """

        # 检查是否有 parent_id 属性（优先级最高）
        if hasattr(data_object, 'parent_id') and data_object.parent_id:
            parent_id = data_object.parent_id
            parent_node = self._find_item_by_id(parent_id)
            if parent_node:
                # 找到父节点，挂载到父节点下
                return parent_node

            else:
                print(f"[警告] 未找到父节点 {parent_id}，挂载到根类别节点")
                return root_parent_node

        if hasattr(data_object, 'model_id'):
            return root_parent_node

        # 曲面对象：始终挂载到"曲面"根节点下，不挂载到剖面下
        if hasattr(data_object, 'surface_id'):
            # 曲面对象直接挂载到曲面根节点
            return root_parent_node

        # 线条对象：挂载到其父剖面节点下
        if hasattr(data_object, 'line_id'):
            # 优先使用 section_id 属性
            if hasattr(data_object, 'section_id') and data_object.section_id:
                section_id = data_object.section_id
                parent_node = self._find_item_by_id(section_id)
                if parent_node:
                    return parent_node

                else:
                    print(f"[警告] 未找到父剖面 {section_id}，将线条添加到根节点")
                    return root_parent_node

            # 如果没有 section_id，挂载到根节点
            else:
                # 线条没有section_id，挂载到根节点
                return root_parent_node

        # 默认返回根类别节点
        return root_parent_node

    def update_fault_info_sync(self, line_id, is_fault, fault_of_surface=None):

        """同步更新断层信息

        Args:
            line_id: 线条ID
            is_fault: 是否为断层
            fault_of_surface: 所属曲面名称

        """

        try:
            line_item = self._find_item_by_id(line_id)
            if line_item and line_item.data_object:
                line_item.data_object.set_fault_info(is_fault, fault_of_surface)
                line_item.refresh_display()
                if self.plotter:
                    self.plotter.render()
            else:
                print(f"[警告] 未找到Line对象: {line_id}")
        except Exception as e:
            print(f"[错误] 断层信息同步失败: {e}")

    def find_line_item(self, line_id):

        """在场景树中查找线条项

        Args:
            line_id: 线条ID
        Returns:
            line_item: 线条节点对象，如果未找到返回None

        """

        try:
            if '剖面' in self.root_nodes:
                section_root = self.root_nodes['剖面']
                for i in range(section_root.childCount()):
                    section_item = section_root.child(i)
                    for j in range(section_item.childCount()):
                        line_item = section_item.child(j)
                        if hasattr(line_item, 'object_id') and line_item.object_id == line_id:
                            return line_item

            return None

        except Exception as e:
            print(f"[SceneManagerPanel] 查找线条项失败: {e}")
            return None

    def get_section_item(self, section_id):

        """根据剖面ID获取剖面节点"""

        for i in range(self.root_nodes["剖面"].childCount()):
            child = self.root_nodes["剖面"].child(i)
            if hasattr(child, 'object_id') and child.object_id == section_id:
                return child

        return None

    def _sync_visibility(self, item, state, direction="both"):

        """递归级联同步引擎

        Parameters:
        -----------
        item : QTreeWidgetItem
            当前操作的节点
        state : bool
            目标状态
        direction : str
            同步方向: "down"（向下分发）, "up"（向上汇总）, "both"（双向）

        """

        try:
            # 阻止信号，防止递归风暴
            self.tree_widget.blockSignals(True)
            if direction in ["down", "both"]:
                # 向下分发：同步所有子节点
                self._cascade_down(item, state)
            if direction in ["up", "both"]:
                # 向上汇总：更新父节点状态
                self._cascade_up(item)
            # 恢复信号
            self.tree_widget.blockSignals(False)
        except Exception as e:
            print(f"同步可见性时出错: {e}")
            # 确保恢复信号
            self.tree_widget.blockSignals(False)

    def _cascade_down(self, parent_item, state):

        """向下级联：强制同步所有子节点"""

        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if isinstance(child, SceneTreeWidgetItem) and child.object_id:
                # 设置子节点状态
                child.setCheckState(0, Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
                # 直接更新业务对象的可见性和actor
                if child.data_object:
                    if hasattr(child.data_object, 'visible'):
                        child.data_object.visible = state
                    # 更新actor可见性
                    if hasattr(child.data_object, 'actor') and child.data_object.actor:
                        if hasattr(child.data_object.actor, 'SetVisibility'):
                            child.data_object.actor.SetVisibility(state)
                    if hasattr(child.data_object, 'actors'):
                        for actor in child.data_object.actors:
                            if hasattr(actor, 'SetVisibility'):
                                actor.SetVisibility(state)
                # 触发子节点的可见性控制
                self.visibilityChanged.emit(child.object_id, state)
                # 递归处理深层子节点
                if child.childCount() > 0:
                    self._cascade_down(child, state)

    def _cascade_up(self, changed_item):

        """向上级联：更新父节点复选框状态（仅UI，不修改业务数据）"""

        parent = changed_item.parent()
        if not parent:
            return

        # 统计子节点状态
        checked_count = 0
        partial_count = 0
        total_count = 0
        for i in range(parent.childCount()):
            child = parent.child(i)
            total_count += 1
            state = child.checkState(0)
            if state == Qt.CheckState.Checked:
                checked_count += 1
            elif state == Qt.CheckState.PartiallyChecked:
                partial_count += 1
        if total_count == 0:
            return
        # 设置父节点状态
        if checked_count == total_count:
            parent.setCheckState(0, Qt.CheckState.Checked)
        elif checked_count == 0 and partial_count == 0:
            parent.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
        # 继续向上（如果父节点还有父节点）
        if parent.parent() is not None:
            self._cascade_up(parent)

    def on_selection_changed(self):

        """处理选中项变化"""

        try:
            selected_items = self.tree_widget.selectedItems()
            # 检查是否在编辑模式下
            current_mode = 'normal' # 默认模式
            if hasattr(self, 'parent_window') and self.parent_window and hasattr(self.parent_window, 'plotter'):
                current_mode = getattr(self.parent_window.plotter, '_current_mode', 'normal')
            # 无论什么模式都更新Wireframe状态
            self.update_wireframe_states()
            if selected_items:
                # 获取选中的第一个项目
                item = selected_items[0]
                if isinstance(item, SceneTreeWidgetItem) and item.object_id:
                    # 发出选中信号
                    self.objectSelected.emit(item.object_id)
        except Exception as e:
            print(f"处理选中变化时出错: {e}")

    def on_item_changed(self, item, column):

        """项目变化处理 - 新架构：直接操作data_object"""

        # 处理根节点（剖面/曲面/块体）的复选框变化
        if column == 0 and not isinstance(item, SceneTreeWidgetItem):
            # 根节点：向下级联所有子节点的可见性
            if item in self.root_nodes.values():
                checked = item.checkState(0) == Qt.CheckState.Checked
                self._sync_visibility(item, checked, direction="down")
                if hasattr(self, 'plotter') and self.plotter:
                    self.plotter.render()
            return
        # 处理复选框变化
        if column == 0 and isinstance(item, SceneTreeWidgetItem):
            checked = item.checkState(0) == Qt.CheckState.Checked
            # 新架构：直接更新业务对象的可见性
            if item.data_object and hasattr(item.data_object, 'visible'):
                item.data_object.visible = checked
                # 如果对象有actor，直接更新actor可见性
                if hasattr(item.data_object, 'actor') and item.data_object.actor:
                    if hasattr(item.data_object.actor, 'SetVisibility'):
                        item.data_object.actor.SetVisibility(checked)
                elif hasattr(item.data_object, 'actors'):
                    # 兼容多个actors的情况
                    for actor in item.data_object.actors:
                        if hasattr(actor, 'SetVisibility'):
                            actor.SetVisibility(checked)
            # 获取对象类型
            object_type = self._get_object_type(item)
            # 判断是否是用户手动操作（通过检查是否有父节点被影响）
            if object_type == "section" and item.childCount() > 0:
                # 父节点操作：向下级联
                self._sync_visibility(item, checked, direction="down")
            else:
                # 子节点操作：向上汇总
                self._sync_visibility(item, checked, direction="up")
            # 发出信号（用于刷新3D视图）
            if item.object_id:
                self.visibilityChanged.emit(item.object_id, checked)
            # 重新渲染视图
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.render()

    def get_item_by_id(self, obj_id):

        """根据ID获取项目"""

        def find_items(parent):

            items = []
            for i in range(parent.childCount()):
                child = parent.child(i)
                if isinstance(child, SceneTreeWidgetItem) and child.object_id == obj_id:
                    items.append(child)
                items.extend(find_items(child))
            return items

        for root in self.root_nodes.values():
            items = find_items(root)
            if items:
                return items[0]

        return None

    def remove_item(self, obj_id):

        """移除项目 - 大管家统筹：同时清理UI节点、3D Actor、内存"""

        item = self.get_item_by_id(obj_id)
        if item:
            # 1. 清理3D Actor（防止幽灵模型）
            if hasattr(item, 'data_object') and item.data_object:
                data_object = item.data_object
                actor = getattr(data_object, 'actor', None)
                if actor and self.plotter:
                    try:
                        self.plotter.remove_actor(actor)
                    except:
                        pass

            # 2. 移除UI树节点
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            return True

        return False

    def set_object_visibility(self, obj_id, visible: bool):

        """设置对象的可见性 - 同时更新业务对象和3D Actor"""

        item = self.get_item_by_id(obj_id)
        if item and hasattr(item, 'data_object') and item.data_object:
            data_object = item.data_object

            # 更新业务对象的visible属性
            if hasattr(data_object, 'visible'):
                data_object.visible = visible

            # 更新3D Actor的可见性
            actor = getattr(data_object, 'actor', None)
            if actor:
                try:
                    actor.SetVisibility(visible)
                except:
                    pass

            # 更新UI节点的显示状态
            if hasattr(item, 'update_visibility_icon'):
                item.update_visibility_icon(visible)

            return True

        return False

    def clear_category(self, category):

        """清空类别下的所有项目"""

        if category in self.root_nodes:
            root = self.root_nodes[category]
            root.takeChildren()

    def get_all_items(self):

        """获取所有场景项"""

        items = []
        for root in self.root_nodes.values():
            for i in range(root.childCount()):
                child = root.child(i)
                if isinstance(child, SceneTreeWidgetItem):
                    items.append(child)
        return items

    def _iter_all_tree_items(self):

        """递归迭代所有树节点（生成器）"""

        for root in self.root_nodes.values():
            yield from self._iter_tree_recursive(root)

    def _iter_tree_recursive(self, parent):

        """递归迭代树节点"""

        for i in range(parent.childCount()):
            child = parent.child(i)
            if isinstance(child, SceneTreeWidgetItem):
                yield child
                # 递归子节点
                if child.childCount() > 0:
                    yield from self._iter_tree_recursive(child)

    # ========== 场景管理方法 ==========

    def set_plotter(self, plotter):

        """设置3D视图引用"""

        self.plotter = plotter

    def get_objects_by_type(self, object_type):

        """获取指定类型的所有对象

        Parameters:
        -----------
        object_type : str
            对象类型: "surface", "block", "section", "line"
        Returns:
        --------
        List[object] : 业务对象列表

        """

        objects = []
        for item in self._iter_all_tree_items():
            if item.data_object:
                item_type = self._get_object_type(item)
                if item_type == object_type:
                    objects.append(item.data_object)
        return objects

    def get_all_objects(self):

        """获取所有业务对象

        Returns:
        --------
        Dict[str, object] : {obj_id: data_object}

        """

        objects = {}
        for item in self._iter_all_tree_items():
            if item.object_id and item.data_object:
                objects[item.object_id] = item.data_object
        return objects

    def on_scene_opacity_changed(self, obj_id, opacity):

        """处理场景对象透明度变化"""

        if self._batch_update_mode:
            return

        try:
            item = self.get_item_by_id(obj_id)
            if item and item.data_object:
                opacity_value = opacity / 100.0
                # 直接设置data_object的透明度
                if hasattr(item.data_object, 'opacity'):
                    item.data_object.opacity = opacity_value
                # 更新actor
                if hasattr(item.data_object, 'actor') and item.data_object.actor:
                    if hasattr(item.data_object.actor, 'GetProperty'):
                        prop = item.data_object.actor.GetProperty()
                        if hasattr(prop, 'SetOpacity'):
                            prop.SetOpacity(opacity_value)
                elif hasattr(item.data_object, 'actors'):
                    for actor in item.data_object.actors:
                        if hasattr(actor, 'GetProperty'):
                            prop = actor.GetProperty()
                            if hasattr(prop, 'SetOpacity'):
                                prop.SetOpacity(opacity_value)
                if self.plotter:
                    self.plotter.render()
        except Exception as e:
            print(f"处理透明度变化时出错: {e}")

    def toggle_wireframe(self, obj_id):

        """切换对象的线框显示模式 - 新架构：优先委托给对象

        Parameters:
        -----------
        obj_id : str
            对象ID

        """

        try:
            # 获取对象的树节点
            item = self.get_item_by_id(obj_id)
            if not item or not item.data_object:
                print(f"未找到对象: {obj_id}")
                return

            data_object = item.data_object
            object_type = self._get_object_type(item)
            if object_type in {"line", "point"}:
                print(f"对象 {obj_id} 是 {object_type} 类型，不需要线框切换")
                return

            # 对于 Surface 和 Block 类型，优先调用对象自身方法
            if hasattr(data_object, 'toggle_wireframe'):
                # 调用对象自身的 toggle_wireframe 方法
                data_object.toggle_wireframe()
                print(f"通过对象自身方法切换 {obj_id} 线框模式")
            elif hasattr(data_object, 'set_render_mode'):
                # 获取当前渲染模式并切换
                current_mode = getattr(data_object, 'render_mode', 'surface')
                new_mode = 'wireframe' if current_mode == 'surface' else 'surface'
                data_object.set_render_mode(new_mode, highlight=True)
                print(f"通过对象方法切换 {obj_id} 到 {new_mode} 模式")
            else:
                # 兜底策略：使用 _set_item_render_mode（兼容旧对象）
                current_mode = 'surface'
                if hasattr(data_object, 'actor') and data_object.actor:
                    actor = data_object.actor
                    if hasattr(actor, 'GetProperty'):
                        prop = actor.GetProperty()
                        current_rep = prop.GetRepresentation()
                        current_mode = 'wireframe' if current_rep == 1 else 'surface'
                    elif hasattr(actor, 'representation'):
                        current_mode = actor.representation
                new_mode = 'surface' if current_mode == 'wireframe' else 'wireframe'
                self._set_item_render_mode(item, new_mode, highlight=True)
                print(f"[警告] 使用兜底策略切换 {obj_id} 到 {new_mode} 模式（建议迁移到新架构）")
            # 重新渲染
            if self.plotter:
                self.plotter.render()
        except Exception as e:
            print(f"切换线框模式失败: {e}")

    def update_wireframe_states(self):

        """更新所有对象的Wireframe状态 - 新架构：遍历树节点，直接操作data_object"""

        try:
            if not hasattr(self, '_outline_actors'):
                self._outline_actors = {}  # {obj_id: outline_actor}
            # 获取当前选中的对象
            selected_items = self.tree_widget.selectedItems()
            selected_ids = set()
            for item in selected_items:
                if isinstance(item, SceneTreeWidgetItem) and item.object_id:
                    selected_ids.add(item.object_id)
            # 获取当前编辑模式
            current_mode = 'normal'
            if hasattr(self, 'parent_window') and self.parent_window and hasattr(self.parent_window, 'plotter'):
                current_mode = getattr(self.parent_window.plotter, '_current_mode', 'normal')
            # 新架构：遍历树中的所有节点，直接操作data_object
            for item in self._iter_all_tree_items():
                if not item.data_object:
                    continue
                obj_id = item.object_id
                is_selected = obj_id in selected_ids
                object_type = self._get_object_type(item)
                # 确定渲染模式
                if current_mode == 'vertex_edit':
                    if object_type == 'block' or (object_type in ['surface', 'section'] and is_selected):
                        render_mode = 'wireframe'
                    else:
                        render_mode = 'surface'
                else:
                    render_mode = 'surface'
                # 优先委托：调用对象的方法设置渲染模式
                if hasattr(item.data_object, 'set_render_mode'):
                    item.data_object.set_render_mode(render_mode, highlight=is_selected)
                else:
                    self._set_item_render_mode(item, render_mode, is_selected)
                # 管理选中轮廓（仅曲面）
                if object_type in ['surface']:
                    self._update_outline_actor(item.data_object, obj_id, is_selected)
            # 重新渲染
            if self.plotter:
                self.plotter.render()
        except Exception as e:
            print(f"更新Wireframe状态时出错: {e}")

    def _update_outline_actor(self, data_object, obj_id, is_selected):
        """管理对象的边缘轮廓actor"""
        try:
            if not self.plotter:
                return
            if is_selected:
                # 已有轮廓actor则跳过
                if obj_id in self._outline_actors:
                    return
                # 获取网格数据
                mesh = None
                if hasattr(data_object, 'mesh') and data_object.mesh is not None:
                    mesh = data_object.mesh
                elif hasattr(data_object, 'actor') and data_object.actor:
                    mapper = data_object.actor.GetMapper()
                    if mapper:
                        mesh_data = mapper.GetInput()
                        if mesh_data:
                            import pyvista as pv
                            mesh = pv.wrap(mesh_data)
                if mesh is None or mesh.n_cells == 0:
                    return
                # 对于体网格（块体），先提取外表面再提取特征边
                import pyvista as pv
                is_volume = not isinstance(mesh, pv.PolyData)
                if is_volume and hasattr(mesh, 'extract_surface'):
                    surface_mesh = mesh.extract_surface()
                else:
                    surface_mesh = mesh
                # 提取边缘轮廓（只提取边界边和特征边，不提取内部网格边）
                # 块体用大角度(80°)只捕获几何棱边，曲面用小角度(30°)捕获更多特征
                angle = 80 if is_volume else 30
                edges = surface_mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                    non_manifold_edges=False,
                    feature_angle=angle
                )
                if edges.n_cells > 0:
                    outline_actor = self.plotter.add_mesh(
                        edges,
                        color='red',
                        line_width=3.0,
                        render_lines_as_tubes=True,
                        name=f"_outline_{obj_id}",
                        pickable=False
                    )
                    self._outline_actors[obj_id] = outline_actor
            else:
                # 移除轮廓actor
                if obj_id in self._outline_actors:
                    try:
                        self.plotter.remove_actor(f"_outline_{obj_id}")
                    except Exception:
                        pass
                    del self._outline_actors[obj_id]
        except Exception as e:
            print(f"更新轮廓actor失败 {obj_id}: {e}")

    def _set_item_render_mode(self, item, mode, highlight=False):

        """设置节点的渲染模式（兼容方法）"""

        if not item.data_object:
            return

        # 获取actor
        actors = []
        if hasattr(item.data_object, 'actor') and item.data_object.actor:
            actors = [item.data_object.actor]
        elif hasattr(item.data_object, 'actors') and item.data_object.actors:
            actors = item.data_object.actors
        # 设置渲染模式
        for actor in actors:
            try:
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if mode == 'wireframe':
                        prop.SetRepresentation(1)
                        if highlight:
                            prop.SetColor(1.0, 0.0, 0.0) # 红色高亮
                            prop.SetLineWidth(2.0)
                    else:
                        prop.SetRepresentation(2)
                        # 恢复原始颜色
                        if hasattr(item.data_object, 'color'):
                            color = item.data_object.color
                            if isinstance(color, (tuple, list)) and len(color) >= 3:
                                prop.SetColor(color[0], color[1], color[2])
                        prop.EdgeVisibilityOff()
                elif hasattr(actor, 'representation'):
                    actor.representation = mode
                    if mode == 'wireframe' and highlight:
                        actor.color = 'red'
                        actor.line_width = 2
                    elif hasattr(item.data_object, 'color'):
                        actor.color = item.data_object.color
            except Exception as e:
                print(f"设置actor渲染模式失败: {e}")

    # ========== 序列化方法 ==========

    def get_scene_tree_data(self) -> List[Dict[str, Any]]:

        """递归遍历场景树结构并提取显示属性

        Returns:
            List[Dict[str, Any]]: 场景对象元数据列表

        """

        try:
            scene_data = []
            for category_name, root_node in self.root_nodes.items():
                # 递归遍历根节点下的所有子节点
                category_data = self._extract_node_data(root_node, category_name)
                scene_data.extend(category_data)
            return scene_data

        except Exception as e:
            return []

    def _extract_node_data(self, node: QTreeWidgetItem, category: str) -> List[Dict[str, Any]]:

        """递归提取节点数据

        Args:
            node: 树节点
            category: 节点类别
        Returns:
            List[Dict[str, Any]]: 节点及其子节点的数据列表

        """

        node_data = []
        for i in range(node.childCount()):
            child = node.child(i)
            if isinstance(child, SceneTreeWidgetItem) and hasattr(child, 'object_id') and child.object_id:
                # 提取节点信息
                item_data = self._extract_item_metadata(child, category)
                if item_data:
                    node_data.append(item_data)
                # 递归处理子节点
                if child.childCount() > 0:
                    child_data = self._extract_node_data(child, category)
                    node_data.extend(child_data)
        return node_data

    def _extract_item_metadata(self, item: SceneTreeWidgetItem, category: str) -> Optional[Dict[str, Any]]:

        """从data_object提取项目元数据"""

        if not hasattr(item, 'data_object') or not item.data_object:
            return None

        obj = item.data_object
        return {

            'id': item.object_id,
            'name': getattr(obj, 'name', ''),
            'type': self._get_object_type(item),
            'color': getattr(obj, 'color', (0.5, 0.5, 0.5)),
            'opacity': getattr(obj, 'opacity', 1.0),
            'visible': getattr(obj, 'visible', True)
        }

    def _type_to_category(self, object_type_str: str) -> str:

        """将对象类型字符串映射到类别名称

        Args:
            object_type_str: 对象类型字符串
        Returns:
            str: 类别名称

        """

        type_mapping = {
            "model": "模型",
            "section": "剖面",
            "surface": "曲面",
            "block": "块体",
            "line": "剖面" # 线条归属于剖面类别
        }
        return type_mapping.get(object_type_str, "模型")

    def clear_all_categories(self):

        """清空所有类别"""

        for category in self.root_nodes.keys():
            self.clear_category(category)

    def get_camera_state(self) -> Optional[Dict[str, Any]]:

        """获取当前相机状态

        Returns:
            Optional[Dict[str, Any]]: 相机状态字典

        """

        try:
            if self.plotter and hasattr(self.plotter, 'camera'):
                camera = self.plotter.camera
                return {

                    'position': list(camera.position),
                    'focal_point': list(camera.focal_point),
                    'view_up': list(camera.view_up),
                    'distance': camera.distance
                }
            return None

        except Exception as e:
            print(f"获取相机状态失败: {e}")
            return None

    def restore_camera_state(self, camera_state: Dict[str, Any]) -> bool:

        """恢复相机状态

        Args:
            camera_state: 相机状态字典
        Returns:
            bool: 是否成功恢复

        """

        try:
            if not self.plotter or not camera_state:
                return False

            camera = self.plotter.camera
            if 'position' in camera_state:
                camera.position = camera_state['position']
            if 'focal_point' in camera_state:
                camera.focal_point = camera_state['focal_point']
            if 'view_up' in camera_state:
                camera.view_up = camera_state['view_up']
            if 'distance' in camera_state:
                camera.distance = camera_state['distance']
            # 重置相机
            self.plotter.reset_camera()
            return True

        except Exception as e:
            print(f"恢复相机状态失败: {e}")
            return False

    def _show_context_menu(self, position):

        """显示右键菜单"""

        # 获取点击的项
        item = self.tree_widget.itemAt(position)
        if not item:
            return

        # 检查是否是根节点
        if item in self.root_nodes.values():
            # 根节点不显示删除菜单
            return

        # 获取对象ID
        obj_id = getattr(item, 'object_id', None)
        if not obj_id:
            return

        # 创建右键菜单
        menu = QMenu(self.tree_widget)
        # 添加删除动作
        delete_action = QAction("删除", self.tree_widget)
        delete_action.triggered.connect(lambda: self._delete_object(obj_id, item))
        menu.addAction(delete_action)
        # 显示菜单
        menu.exec(self.tree_widget.viewport().mapToGlobal(position))

    def _delete_object(self, obj_id: str, item):

        """删除对象 - 极简主义：只做两件事

        Parameters:
        -----------
        obj_id : str
            对象ID
        item : QTreeWidgetItem
            树节点

        """

        try:
            print(f"\n=== 删除对象: {obj_id} ===")
            # 第一步：调用业务对象的cleanup方法
            if isinstance(item, SceneTreeWidgetItem) and item.data_object:
                if hasattr(item.data_object, 'cleanup'):
                    item.data_object.cleanup(self.plotter)
                    print(f"✓ 对象已自清理")
                else:
                    print(f"⚠ 对象 {type(item.data_object)} 没有cleanup方法")
            # 第二步：从场景树中移除节点
            parent = item.parent()
            if parent:
                parent.removeChild(item)
                print(f"✓ 已从场景树移除节点")
            else:
                print(f"⚠ 节点没有父节点，无法移除")
            print(f"✓ 对象删除完成")
        except Exception as e:
            print(f"✗ 删除对象失败: {e}")
            traceback.print_exc()

    # ========== 清理完成，所有操作统一通过新架构 ==========
    
    def update_scene_panel(self):
        """更新场景面板
        
        重新加载所有对象并更新显示，保持展开状态
        """
        try:
            print("\n=== 更新场景面板 ===")
            
            # 保存当前展开状态（包括剖面节点的展开状态）
            expanded_categories = {}
            expanded_sections = {}
            for category, root_node in self.root_nodes.items():
                expanded_categories[category] = root_node.isExpanded()
                
                # 如果是剖面类别，保存每个剖面的展开状态
                if category == "剖面":
                    for i in range(root_node.childCount()):
                        section_item = root_node.child(i)
                        if hasattr(section_item, 'object_id'):
                            expanded_sections[section_item.object_id] = section_item.isExpanded()
            
            # 收集所有现有对象（包括剖面的子对象）
            all_objects = []
            for category in ["剖面", "曲面", "块体"]:
                root_node = self.root_nodes.get(category)
                if root_node:
                    for i in range(root_node.childCount()):
                        child = root_node.child(i)
                        if isinstance(child, SceneTreeWidgetItem) and child.data_object:
                            obj_info = {
                                'category': category,
                                'object': child.data_object,
                                'object_id': child.object_id
                            }
                            
                            # 如果是剖面，收集其子对象（线条）
                            if category == "剖面" and child.childCount() > 0:
                                child_lines = []
                                for j in range(child.childCount()):
                                    line_child = child.child(j)
                                    if isinstance(line_child, SceneTreeWidgetItem) and line_child.data_object:
                                        child_lines.append({
                                            'object': line_child.data_object,
                                            'object_id': line_child.object_id
                                        })
                                obj_info['children'] = child_lines
                            
                            all_objects.append(obj_info)
            
            print(f"找到 {len(all_objects)} 个对象需要更新")
            
            # 清空所有根节点的子节点
            for root_node in self.root_nodes.values():
                root_node.takeChildren()
            
            # 重新添加所有对象
            for obj_data in all_objects:
                try:
                    category = obj_data['category']
                    data_object = obj_data['object']
                    
                    # 重新添加到场景树
                    self.add_object(data_object, category)
                    
                    # 如果是剖面且有子对象，重新添加子对象
                    if category == "剖面" and 'children' in obj_data:
                        for child_info in obj_data['children']:
                            child_obj = child_info['object']
                            # 添加线条到剖面
                            self.add_object(child_obj, category)
                    
                except Exception as e:
                    print(f"⚠ 重新添加对象失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 恢复展开状态
            for category, is_expanded in expanded_categories.items():
                if category in self.root_nodes:
                    self.root_nodes[category].setExpanded(is_expanded)
            
            # 恢复剖面节点的展开状态
            if "剖面" in self.root_nodes:
                section_root = self.root_nodes["剖面"]
                for i in range(section_root.childCount()):
                    section_item = section_root.child(i)
                    if hasattr(section_item, 'object_id'):
                        section_id = section_item.object_id
                        if section_id in expanded_sections:
                            section_item.setExpanded(expanded_sections[section_id])
            
            # 刷新3D视图
            if self.plotter:
                self.plotter.render()
            
            print(f"✓ 场景面板更新完成，共 {len(all_objects)} 个对象")
            
        except Exception as e:
            print(f"✗ 更新场景面板失败: {e}")
            import traceback
            traceback.print_exc()

