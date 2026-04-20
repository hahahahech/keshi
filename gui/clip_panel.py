"""
裁剪控制面板（仅保留不规则裁剪）。
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ClipPanel(QWidget):
    maskDrawingStartRequested = pyqtSignal(str, object)
    maskDrawingCancelRequested = pyqtSignal()
    maskClipRequested = pyqtSignal(str)
    clearDerivedRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene_object = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QFormLayout()
        self.target_label = QLabel("未选择")
        header.addRow("目标对象", self.target_label)
        layout.addLayout(header)

        self.mask_box = QGroupBox("不规则裁剪（按绘制边界保留内部）")
        mask_layout = QGridLayout(self.mask_box)
        self.mask_plane_combo = QComboBox(self)
        self.mask_plane_combo.addItem("XOY（俯视）", "xoy")
        self.mask_plane_combo.addItem("XOZ（前视）", "xoz")
        self.mask_plane_combo.addItem("YOZ（侧视）", "yoz")
        self.mask_state_label = QLabel("未开始绘制")
        self.mask_count_label = QLabel("当前点数：0")
        self.start_mask_button = QPushButton("开始绘制边界")
        self.cancel_mask_button = QPushButton("取消绘制")
        self.apply_mask_button = QPushButton("应用不规则裁剪")
        self.start_mask_button.clicked.connect(self._emit_start_mask_drawing)
        self.cancel_mask_button.clicked.connect(self._emit_cancel_mask_drawing)
        self.apply_mask_button.clicked.connect(self._emit_mask_clip)
        mask_layout.addWidget(QLabel("绘制平面"), 0, 0)
        mask_layout.addWidget(self.mask_plane_combo, 0, 1)
        mask_layout.addWidget(self.mask_state_label, 1, 0, 1, 2)
        mask_layout.addWidget(self.mask_count_label, 2, 0, 1, 2)
        mask_layout.addWidget(self.start_mask_button, 3, 0)
        mask_layout.addWidget(self.cancel_mask_button, 3, 1)
        mask_layout.addWidget(self.apply_mask_button, 4, 0, 1, 2)
        layout.addWidget(self.mask_box)

        actions_layout = QHBoxLayout()
        self.clear_derived_button = QPushButton("清除派生裁剪")
        self.clear_derived_button.clicked.connect(self._emit_clear)
        actions_layout.addWidget(self.clear_derived_button)
        layout.addLayout(actions_layout)

        layout.addStretch(1)
        self._set_enabled(False)

    def _set_enabled(self, enabled: bool):
        self.mask_box.setEnabled(enabled)
        self.start_mask_button.setEnabled(enabled)
        self.cancel_mask_button.setEnabled(False)
        self.apply_mask_button.setEnabled(False)
        self.clear_derived_button.setEnabled(enabled)
        self.mask_state_label.setText("未开始绘制")
        self.mask_count_label.setText("当前点数：0")

    def set_scene_object(self, scene_object):
        self._scene_object = scene_object
        if scene_object is None:
            self.target_label.setText("未选择")
            self._set_enabled(False)
            return
        self.target_label.setText(scene_object.name)
        self._set_enabled(True)

    def set_mask_state(self, drawing: bool, point_count: int):
        if drawing:
            self.mask_state_label.setText("绘制中：左键加点，右键撤销，双击完成")
        else:
            self.mask_state_label.setText("未开始绘制" if point_count == 0 else "边界已完成，可应用裁剪")
        self.mask_count_label.setText(f"当前点数：{int(point_count)}")
        self.start_mask_button.setEnabled(self._scene_object is not None and not drawing)
        self.cancel_mask_button.setEnabled(self._scene_object is not None and (drawing or point_count > 0))
        self.apply_mask_button.setEnabled(self._scene_object is not None and point_count >= 3)

    def _mask_draw_value_from_bounds(self, draw_plane: str | None) -> float:
        if self._scene_object is None or self._scene_object.bounds is None:
            return 0.0
        bounds = self._scene_object.bounds
        plane = str(draw_plane or "xoy").lower()
        if plane == "xoz":
            return float(bounds[3])
        if plane == "yoz":
            return float(bounds[1])
        return float(bounds[5])

    def _emit_start_mask_drawing(self):
        if self._scene_object is None:
            return
        draw_plane = self.mask_plane_combo.currentData()
        draw_value = self._mask_draw_value_from_bounds(draw_plane)
        self.maskDrawingStartRequested.emit(
            self._scene_object.object_id,
            {"draw_plane": draw_plane, "draw_value": draw_value},
        )

    def _emit_cancel_mask_drawing(self):
        self.maskDrawingCancelRequested.emit()

    def _emit_mask_clip(self):
        if self._scene_object is None:
            return
        self.maskClipRequested.emit(self._scene_object.object_id)

    def _emit_clear(self):
        if self._scene_object is None:
            return
        self.clearDerivedRequested.emit(self._scene_object.object_id)

    def focus_clip_controls(self):
        self.mask_box.setFocus()
        self.start_mask_button.setFocus()
