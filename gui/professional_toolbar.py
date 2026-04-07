"""
可视化工作区主工具栏。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QToolBar, QWidget


class ProfessionalToolbar(QToolBar):
    action_triggered = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("可视化工具栏")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.actions_dict: dict[str, QAction] = {}
        self._init_toolbar()

    def _init_toolbar(self):
        self._add_action("import_data", "导入数据")
        self._add_action("open_project", "打开工程")
        self._add_action("save_project", "保存工程")
        self.addSeparator()
        self._add_action("reset_view", "重置视图")
        self._add_action("clear_derived", "清除派生对象")
        self._add_action("open_slice_window", "切片窗口")
        self._add_action("open_clip_window", "裁剪窗口")
        self.addSeparator()
        self._add_action("export_screenshot", "导出截图")
        self._add_action("export_object", "导出数据")

    def _add_action(self, name: str, text: str) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(
            lambda checked=False, action_name=name: self.action_triggered.emit(action_name)
        )
        self.addAction(action)
        self.actions_dict[name] = action
        return action

    def set_action_enabled(self, action_name: str, enabled: bool):
        if action_name in self.actions_dict:
            self.actions_dict[action_name].setEnabled(enabled)
