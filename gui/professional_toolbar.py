"""
纯可视化工具栏组件
"""

import os
from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QStyle, QToolBar, QWidget


class ProfessionalToolbar(QToolBar):
    """面向可视化浏览场景的简化工具栏。"""

    action_triggered = pyqtSignal(str)
    mode_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("可视化工具栏")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        self.icon_path = os.path.join(os.path.dirname(__file__), "icons")
        self.actions_dict: Dict[str, QAction] = {}
        self.parent_window = None

        self._init_toolbar()

    def _init_toolbar(self):
        self._add_action(
            name="import_model",
            text="导入模型",
            tooltip="导入通用三维模型文件",
            icon_name="open.png",
        )
        self.addSeparator()
        self._add_action(
            name="reset_view",
            text="重置视图",
            tooltip="重置三维视图到初始位置",
            icon_name="reset.png",
        )
        self._add_action(
            name="toggle_axes",
            text="方向组件",
            tooltip="显示或隐藏右上角方向组件",
            icon_name="property.png",
        )
        self._add_action(
            name="toggle_grid",
            text="网格",
            tooltip="显示或隐藏参考网格",
            icon_name="manage.png",
        )
        self._add_action(
            name="toggle_origin_axes",
            text="原点坐标轴",
            tooltip="显示或隐藏原点坐标轴",
            icon_name="stratum.png",
        )
        self._add_action(
            name="toggle_axis_scales",
            text="坐标轴刻度",
            tooltip="显示或隐藏坐标轴刻度",
            icon_name="settings.png",
        )
        self.addSeparator()
        self._add_action(
            name="clear_section_outline",
            text="清除轮廓",
            tooltip="清除当前高亮轮廓",
            icon_name="clip.png",
        )
        self._add_action(
            name="export_screenshot",
            text="导出截图",
            tooltip="导出当前视图截图",
            icon_name="export.png",
        )

    def _add_action(self, name: str, text: str, tooltip: str, icon_name: Optional[str] = None) -> QAction:
        action = QAction(text, self)
        action.setToolTip(tooltip)

        if icon_name:
            icon = self._get_icon(icon_name)
            if icon and not icon.isNull():
                action.setIcon(icon)

        action.triggered.connect(lambda checked=False, action_name=name: self.action_triggered.emit(action_name))
        self.addAction(action)
        self.actions_dict[name] = action
        return action

    def _get_icon(self, icon_name: str) -> QIcon:
        icon_file = os.path.join(self.icon_path, icon_name)
        if os.path.exists(icon_file):
            return QIcon(icon_file)

        icon_map = {
            "open.png": "SP_DialogOpenButton",
            "reset.png": "SP_BrowserReload",
            "property.png": "SP_FileDialogDetailedView",
            "manage.png": "SP_FileDialogListView",
            "stratum.png": "SP_ArrowUp",
            "settings.png": "SP_FileDialogDetailedView",
            "clip.png": "SP_DialogDiscardButton",
            "export.png": "SP_DialogSaveButton",
        }

        standard_icon = icon_map.get(icon_name)
        if not standard_icon:
            return QIcon()

        style = self.style()
        enum_value = getattr(QStyle.StandardPixmap, standard_icon, None)
        if enum_value is None:
            return QIcon()
        return style.standardIcon(enum_value)

    def set_parent_window(self, parent_window):
        self.parent_window = parent_window

    def set_action_enabled(self, action_name: str, enabled: bool):
        if action_name in self.actions_dict:
            self.actions_dict[action_name].setEnabled(enabled)
