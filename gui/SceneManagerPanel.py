"""
可视化工作区的场景与数据集树面板。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QAction


ROOT_LABELS = {
    "dataset": "数据集",
    "slice": "切片",
    "section": "剖面",
    "isosurface": "等值面",
    "clip": "裁剪",
    "helper": "辅助对象",
}

OBJECT_TYPE_LABELS = {
    "dataset": "数据集",
    "slice": "切片",
    "section": "剖面",
    "isosurface": "等值面",
    "clip": "裁剪",
    "helper": "辅助对象",
}


class OpacityEditor(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0.0, 100.0)
        self.setSingleStep(5.0)
        self.setDecimals(0)
        self.setSuffix("%")
        self.setMaximumWidth(72)


class SceneTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, scene_object, parent=None):
        super().__init__(parent)
        self.data_object = scene_object
        self.object_id = scene_object.object_id
        self.opacity_editor: OpacityEditor | None = None
        self.refresh()

    def refresh(self):
        scene_object = self.data_object
        self.setText(0, scene_object.name)
        self.setText(1, OBJECT_TYPE_LABELS.get(scene_object.object_type, scene_object.object_type))
        self.setCheckState(
            0,
            Qt.CheckState.Checked if scene_object.visible else Qt.CheckState.Unchecked,
        )
        self.setToolTip(0, scene_object.object_id)


class SceneManagerPanel(QWidget):
    visibilityChanged = pyqtSignal(str, bool)
    opacityChanged = pyqtSignal(str, float)
    objectSelected = pyqtSignal(str)
    objectDeselected = pyqtSignal(str)
    renameRequested = pyqtSignal(str, str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.root_nodes: dict[str, QTreeWidgetItem] = {}
        self._items_by_id: dict[str, SceneTreeWidgetItem] = {}
        self.plotter = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree_widget = QTreeWidget(self)
        self.tree_widget.setHeaderLabels(["对象", "类型", "透明度"])
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree_widget)

        for object_type, label in ROOT_LABELS.items():
            root = QTreeWidgetItem(self.tree_widget, [label, "", ""])
            root.setFlags(Qt.ItemFlag.ItemIsEnabled)
            root.setExpanded(True)
            self.root_nodes[object_type] = root

    def set_plotter(self, plotter):
        self.plotter = plotter

    def add_object(self, scene_object, category=None):
        root_type = category or scene_object.object_type
        root = self.root_nodes.get(root_type, self.root_nodes["helper"])

        previous_state = self._updating
        self._updating = True
        item = SceneTreeWidgetItem(scene_object, root)
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
        )
        opacity_editor = OpacityEditor(self.tree_widget)
        opacity_editor.setValue(scene_object.opacity * 100.0)
        opacity_editor.valueChanged.connect(
            lambda value, object_id=scene_object.object_id: self.opacityChanged.emit(
                object_id, value / 100.0
            )
        )
        item.opacity_editor = opacity_editor
        self.tree_widget.setItemWidget(item, 2, opacity_editor)
        self._items_by_id[scene_object.object_id] = item
        root.setExpanded(True)
        self._updating = previous_state
        return item

    def rebuild(self, scene_objects):
        self._updating = True
        self.clear_all_categories()
        for scene_object in scene_objects:
            self.add_object(scene_object)
        self._updating = False

    def get_item_by_id(self, object_id: str) -> SceneTreeWidgetItem | None:
        return self._items_by_id.get(object_id)

    def refresh_object(self, scene_object):
        item = self.get_item_by_id(scene_object.object_id)
        if item is None:
            return self.add_object(scene_object)
        previous_state = self._updating
        self._updating = True
        item.data_object = scene_object
        item.refresh()
        if item.opacity_editor is not None:
            item.opacity_editor.setValue(scene_object.opacity * 100.0)
        self._updating = previous_state
        return item

    def remove_object(self, object_id: str):
        item = self._items_by_id.pop(object_id, None)
        if item is None:
            return
        parent = item.parent()
        if parent is not None:
            parent.removeChild(item)

    def clear_all_categories(self):
        self._items_by_id.clear()
        for root in self.root_nodes.values():
            root.takeChildren()

    def on_scene_opacity_changed(self, object_id: str, opacity: float):
        item = self.get_item_by_id(object_id)
        if item is None:
            return
        item.data_object.opacity = opacity
        if item.opacity_editor is not None:
            self._updating = True
            item.opacity_editor.setValue(opacity * 100.0)
            self._updating = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._updating or not isinstance(item, SceneTreeWidgetItem):
            return
        if column == 0:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.visibilityChanged.emit(item.object_id, visible)
            if item.text(0) != item.data_object.name:
                self.renameRequested.emit(item.object_id, item.text(0))

    def _on_selection_changed(self):
        selected = self.tree_widget.selectedItems()
        if not selected:
            self.objectDeselected.emit("")
            return
        item = selected[0]
        if isinstance(item, SceneTreeWidgetItem):
            self.objectSelected.emit(item.object_id)

    def _show_context_menu(self, position):
        item = self.tree_widget.itemAt(position)
        if not isinstance(item, SceneTreeWidgetItem):
            return

        menu = QMenu(self.tree_widget)
        delete_action = QAction("删除", self.tree_widget)
        delete_action.triggered.connect(lambda: self.deleteRequested.emit(item.object_id))
        menu.addAction(delete_action)
        menu.exec(self.tree_widget.viewport().mapToGlobal(position))
