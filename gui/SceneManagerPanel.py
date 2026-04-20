"""
可视化工作区中的对象树面板。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QInputDialog,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


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


OBJECT_TYPE_LABELS["drillhole"] = "钻孔数据"


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
    openPropertyRequested = pyqtSignal(str)
    sliceMoveRequested = pyqtSignal(str)
    sliceTiltRequested = pyqtSignal(str)

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
        # 只允许通过双击“名称列”触发编辑，其他列完全不可编辑
        self.tree_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
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
        root_alias = {
            "section": "slice",
            "drillhole": "dataset",
        }
        inferred_type = root_alias.get(scene_object.object_type, scene_object.object_type)
        root_type = category or inferred_type
        root = self.root_nodes.get(root_type, self.root_nodes["helper"])

        previous_state = self._updating
        self._updating = True
        item = SceneTreeWidgetItem(scene_object, root)
        item.setFlags(
            item.flags()
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
        opacity_editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        opacity_editor.customContextMenuRequested.connect(
            lambda pos, target_item=item, editor=opacity_editor: self._show_item_context_menu(
                target_item,
                editor.mapToGlobal(pos),
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
        if column != 0:
            return

        visible = item.checkState(0) == Qt.CheckState.Checked
        current_visible = bool(getattr(item.data_object, "visible", True))
        edited_name = item.text(0).strip()
        current_name = str(getattr(item.data_object, "name", "") or "")

        # 先处理重命名，避免可见性刷新把新名称覆盖回旧值
        if not edited_name:
            self._updating = True
            item.setText(0, current_name)
            self._updating = False
        elif edited_name != current_name:
            item.data_object.name = edited_name
            self.renameRequested.emit(item.object_id, edited_name)

        if visible != current_visible:
            item.data_object.visible = visible
            self.visibilityChanged.emit(item.object_id, visible)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        if self._updating or not isinstance(item, SceneTreeWidgetItem):
            return
        # 仅名称列允许编辑，类型列双击不进入编辑
        if column == 0:
            self.tree_widget.editItem(item, 0)

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
            selected = self.tree_widget.selectedItems()
            if not selected or not isinstance(selected[0], SceneTreeWidgetItem):
                return
            item = selected[0]
        self._show_item_context_menu(item, self.tree_widget.viewport().mapToGlobal(position))

    def _show_item_context_menu(self, item: SceneTreeWidgetItem, global_pos):
        menu = QMenu(self.tree_widget)
        object_type = getattr(item.data_object, "object_type", "")
        params = dict(getattr(item.data_object, "parameters", {}) or {})
        kind = str(params.get("kind") or "").lower()
        is_slice_like = object_type == "slice" or kind in {"axis", "orthogonal", "plane"}

        if object_type in {"dataset", "drillhole"}:
            open_property_action = QAction("打开属性控制", self.tree_widget)
            open_property_action.triggered.connect(
                lambda: self.openPropertyRequested.emit(item.object_id)
            )
            menu.addAction(open_property_action)

        rename_action = QAction("重命名", self.tree_widget)
        rename_action.triggered.connect(lambda: self._rename_item(item))
        menu.addAction(rename_action)

        if is_slice_like:
            move_slice_action = QAction("平移切片", self.tree_widget)
            move_slice_action.triggered.connect(lambda: self.sliceMoveRequested.emit(item.object_id))
            menu.addAction(move_slice_action)

            tilt_slice_action = QAction("倾斜切片", self.tree_widget)
            tilt_slice_action.triggered.connect(lambda: self.sliceTiltRequested.emit(item.object_id))
            menu.addAction(tilt_slice_action)

        delete_action = QAction("删除", self.tree_widget)
        delete_action.triggered.connect(lambda: self.deleteRequested.emit(item.object_id))
        menu.addAction(delete_action)
        menu.exec(global_pos)

    def _rename_item(self, item: SceneTreeWidgetItem):
        current_name = str(getattr(item.data_object, "name", "") or "")
        new_name, ok = QInputDialog.getText(self, "重命名对象", "请输入新名称：", text=current_name)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if new_name == current_name:
            return

        self._updating = True
        item.setText(0, new_name)
        self._updating = False
        item.data_object.name = new_name
        self.renameRequested.emit(item.object_id, new_name)
