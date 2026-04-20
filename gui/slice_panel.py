"""
切片控制面板。
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class SlicePanel(QWidget):
    axisBatchSliceRequested = pyqtSignal(str, object)
    orthogonalSliceRequested = pyqtSignal(str, object)
    planeSliceRequested = pyqtSignal(str, object)
    polylineDrawingRequested = pyqtSignal(str, object)
    polylineDrawingCancelled = pyqtSignal()
    polylineSectionRequested = pyqtSignal(str, object)
    clearDerivedRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene_object = None
        self._current_mode = "orthogonal"
        self._polyline_point_count = 0
        self.mode_checkboxes: dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QFormLayout()
        self.target_label = QLabel("未选择")
        header.addRow("目标对象", self.target_label)
        layout.addLayout(header)

        mode_box = QGroupBox("切片模式")
        mode_layout = QHBoxLayout(mode_box)
        for mode, label in (
            ("orthogonal", "三向切片"),
            ("step", "步长切片"),
            ("plane", "平面切片"),
            ("section", "折线剖面"),
        ):
            checkbox = QCheckBox(label, self)
            checkbox.toggled.connect(lambda checked, key=mode: self._on_mode_toggled(key, checked))
            mode_layout.addWidget(checkbox)
            self.mode_checkboxes[mode] = checkbox
        layout.addWidget(mode_box)

        self.mode_stack = QStackedWidget(self)
        self.mode_stack.addWidget(self._build_orthogonal_page())
        self.mode_stack.addWidget(self._build_step_page())
        self.mode_stack.addWidget(self._build_plane_page())
        self.mode_stack.addWidget(self._build_section_page())
        layout.addWidget(self.mode_stack)

        actions_layout = QHBoxLayout()
        self.reset_controls_button = QPushButton("重置参数")
        self.clear_derived_button = QPushButton("清除派生切片")
        self.reset_controls_button.clicked.connect(self._reset_from_bounds)
        self.clear_derived_button.clicked.connect(self._emit_clear)
        actions_layout.addWidget(self.reset_controls_button)
        actions_layout.addWidget(self.clear_derived_button)
        layout.addLayout(actions_layout)

        layout.addStretch(1)
        self._set_mode("orthogonal")
        self._set_enabled(False)

    def _build_orthogonal_page(self) -> QWidget:
        page = QWidget(self)
        layout = QGridLayout(page)
        self.orthogonal_spins = {axis: self._make_float_spinbox() for axis in ("x", "y", "z")}
        for row, axis in enumerate(("x", "y", "z")):
            layout.addWidget(QLabel(f"{axis.upper()} 坐标"), row, 0)
            layout.addWidget(self.orthogonal_spins[axis], row, 1)
        self.create_orthogonal_button = QPushButton("生成三向切片")
        self.create_orthogonal_button.clicked.connect(self._emit_orthogonal_slice)
        layout.addWidget(self.create_orthogonal_button, 3, 0, 1, 2)
        return page

    def _build_step_page(self) -> QWidget:
        page = QWidget(self)
        layout = QGridLayout(page)
        self.step_axis_combo = QComboBox(self)
        self.step_axis_combo.addItem("沿 X 方向", "x")
        self.step_axis_combo.addItem("沿 Y 方向", "y")
        self.step_axis_combo.addItem("沿 Z 方向", "z")
        self.step_axis_combo.currentIndexChanged.connect(self._sync_step_range_to_axis)
        self.step_start_spin = self._make_float_spinbox()
        self.step_end_spin = self._make_float_spinbox()
        self.step_size_spin = self._make_float_spinbox(default=10.0, minimum=0.000001)
        layout.addWidget(QLabel("切片方向"), 0, 0)
        layout.addWidget(self.step_axis_combo, 0, 1)
        layout.addWidget(QLabel("起点"), 1, 0)
        layout.addWidget(self.step_start_spin, 1, 1)
        layout.addWidget(QLabel("终点"), 2, 0)
        layout.addWidget(self.step_end_spin, 2, 1)
        layout.addWidget(QLabel("步长"), 3, 0)
        layout.addWidget(self.step_size_spin, 3, 1)
        self.create_step_button = QPushButton("按步长批量生成")
        self.create_step_button.clicked.connect(self._emit_step_slice_batch)
        layout.addWidget(self.create_step_button, 4, 0, 1, 2)
        return page

    def _build_plane_page(self) -> QWidget:
        page = QWidget(self)
        layout = QGridLayout(page)
        self.origin_spins = {axis: self._make_float_spinbox() for axis in ("x", "y", "z")}
        self.normal_spins = {
            axis: self._make_float_spinbox(default=1.0 if axis == "z" else 0.0)
            for axis in ("x", "y", "z")
        }
        for row, axis in enumerate(("x", "y", "z")):
            layout.addWidget(QLabel(f"原点 {axis.upper()}"), row, 0)
            layout.addWidget(self.origin_spins[axis], row, 1)
            layout.addWidget(QLabel(f"法向 {axis.upper()}"), row, 2)
            layout.addWidget(self.normal_spins[axis], row, 3)
        self.create_plane_button = QPushButton("生成平面切片")
        self.create_plane_button.clicked.connect(self._emit_plane_slice)
        layout.addWidget(self.create_plane_button, 3, 0, 1, 4)
        return page

    def _build_section_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        status_box = QGroupBox("折线绘制")
        status_layout = QGridLayout(status_box)
        self.section_state_label = QLabel("未开始绘制")
        self.section_count_label = QLabel("当前点数：0")
        self.start_polyline_button = QPushButton("开始绘制")
        self.cancel_polyline_button = QPushButton("取消绘制")
        self.start_polyline_button.clicked.connect(self._emit_start_polyline_drawing)
        self.cancel_polyline_button.clicked.connect(self._emit_cancel_polyline_drawing)
        status_layout.addWidget(self.section_state_label, 0, 0, 1, 2)
        status_layout.addWidget(self.section_count_label, 1, 0, 1, 2)
        status_layout.addWidget(self.start_polyline_button, 2, 0)
        status_layout.addWidget(self.cancel_polyline_button, 2, 1)
        layout.addWidget(status_box)

        param_box = QGroupBox("平面参数")
        param_layout = QGridLayout(param_box)
        self.section_plane_combo = QComboBox(self)
        self.section_plane_combo.addItem("XOY（俯视）", "xoy")
        self.section_plane_combo.addItem("XOZ（前视）", "xoz")
        self.section_plane_combo.addItem("YOZ（侧视）", "yoz")
        self.section_plane_combo.currentIndexChanged.connect(self._sync_section_range_for_draw_plane)
        self.section_top_spin = self._make_float_spinbox()
        self.section_bottom_spin = self._make_float_spinbox()
        self.section_step_spin = self._make_float_spinbox(default=25.0, minimum=0.000001)
        self.section_vertical_spin = QSpinBox(self)
        self.section_vertical_spin.setRange(2, 400)
        self.section_vertical_spin.setValue(20)
        self.section_top_label = QLabel("上界 Z")
        self.section_bottom_label = QLabel("下界 Z")
        param_layout.addWidget(QLabel("绘制平面"), 0, 0)
        param_layout.addWidget(self.section_plane_combo, 0, 1)
        param_layout.addWidget(self.section_top_label, 1, 0)
        param_layout.addWidget(self.section_top_spin, 1, 1)
        param_layout.addWidget(self.section_bottom_label, 2, 0)
        param_layout.addWidget(self.section_bottom_spin, 2, 1)
        param_layout.addWidget(QLabel("沿线步长"), 3, 0)
        param_layout.addWidget(self.section_step_spin, 3, 1)
        param_layout.addWidget(QLabel("垂向层数"), 4, 0)
        param_layout.addWidget(self.section_vertical_spin, 4, 1)
        layout.addWidget(param_box)

        self.create_section_button = QPushButton("生成折线剖面")
        self.create_section_button.clicked.connect(self._emit_polyline_section)
        layout.addWidget(self.create_section_button)
        layout.addStretch(1)
        return page

    def _make_float_spinbox(
        self,
        default: float = 0.0,
        minimum: float = -1e12,
        maximum: float = 1e12,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setDecimals(6)
        spin.setValue(default)
        return spin

    def _set_enabled(self, enabled: bool):
        widgets = [
            self.create_orthogonal_button,
            self.step_axis_combo,
            self.step_start_spin,
            self.step_end_spin,
            self.step_size_spin,
            self.create_step_button,
            self.create_plane_button,
            self.reset_controls_button,
            self.clear_derived_button,
            self.start_polyline_button,
            self.cancel_polyline_button,
            self.section_plane_combo,
            self.section_top_spin,
            self.section_bottom_spin,
            self.section_step_spin,
            self.section_vertical_spin,
            self.create_section_button,
        ]
        widgets.extend(self.orthogonal_spins.values())
        widgets.extend(self.origin_spins.values())
        widgets.extend(self.normal_spins.values())
        for widget in widgets:
            widget.setEnabled(enabled)
        for checkbox in self.mode_checkboxes.values():
            checkbox.setEnabled(enabled)
        self._update_section_buttons()

    def _set_mode(self, mode: str):
        self._current_mode = mode
        order = {"orthogonal": 0, "step": 1, "plane": 2, "section": 3}
        for key, checkbox in self.mode_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key == mode)
            checkbox.blockSignals(False)
        self.mode_stack.setCurrentIndex(order[mode])

    def _on_mode_toggled(self, mode: str, checked: bool):
        if checked:
            self._set_mode(mode)
            return
        if self._current_mode == mode:
            self.mode_checkboxes[mode].blockSignals(True)
            self.mode_checkboxes[mode].setChecked(True)
            self.mode_checkboxes[mode].blockSignals(False)

    def set_scene_object(self, scene_object):
        self._scene_object = scene_object
        self._polyline_point_count = 0
        self.set_polyline_state(False, 0)
        if scene_object is None:
            self.target_label.setText("未选择")
            self._set_enabled(False)
            return
        self.target_label.setText(scene_object.name)
        self._set_enabled(True)
        self._reset_from_bounds()

    def set_polyline_state(self, drawing: bool, point_count: int):
        self._polyline_point_count = int(point_count)
        if drawing:
            self.section_state_label.setText("绘制中：左键加点，右键撤销，双击完成")
        else:
            self.section_state_label.setText("未开始绘制" if point_count == 0 else "折线已完成，可直接生成平面切片")
        self.section_count_label.setText(f"当前点数：{point_count}")
        self._update_section_buttons(drawing=drawing)

    def _update_section_buttons(self, drawing: bool | None = None):
        if drawing is None:
            drawing = self.section_state_label.text().startswith("绘制中")
        has_object = self._scene_object is not None
        self.start_polyline_button.setEnabled(has_object)
        self.cancel_polyline_button.setEnabled(has_object and (drawing or self._polyline_point_count > 0))
        self.create_section_button.setEnabled(has_object and self._polyline_point_count >= 2)

    def _reset_from_bounds(self):
        if self._scene_object is None:
            return
        bounds = self._scene_object.bounds
        centers = {
            "x": (bounds[0] + bounds[1]) / 2.0,
            "y": (bounds[2] + bounds[3]) / 2.0,
            "z": (bounds[4] + bounds[5]) / 2.0,
        }
        for axis, value in centers.items():
            self.orthogonal_spins[axis].setValue(float(value))
            self.origin_spins[axis].setValue(float(value))
        self.section_plane_combo.setCurrentIndex(0)
        self._sync_section_range_for_draw_plane()
        horizontal_span = max(abs(bounds[1] - bounds[0]), abs(bounds[3] - bounds[2]), 1.0)
        section_low, section_high = self._draw_plane_section_bounds(self.section_plane_combo.currentData())
        vertical_span = max(abs(section_high - section_low), 1.0)
        self.section_step_spin.setValue(horizontal_span / 20.0)
        self.section_vertical_spin.setValue(max(10, min(80, int(round(vertical_span / max(horizontal_span / 20.0, 1.0))))))
        self._sync_step_range_to_axis()

    def _draw_plane_section_bounds(self, draw_plane: str | None) -> tuple[float, float]:
        if self._scene_object is None or self._scene_object.bounds is None:
            return 0.0, 0.0
        bounds = self._scene_object.bounds
        plane = str(draw_plane or "xoy").lower()
        if plane == "xoz":
            return float(bounds[2]), float(bounds[3])
        if plane == "yoz":
            return float(bounds[0]), float(bounds[1])
        return float(bounds[4]), float(bounds[5])

    def _draw_plane_fixed_axis_label(self, draw_plane: str | None) -> str:
        plane = str(draw_plane or "xoy").lower()
        if plane == "xoz":
            return "Y"
        if plane == "yoz":
            return "X"
        return "Z"

    def _sync_section_range_for_draw_plane(self):
        if self._scene_object is None:
            return
        draw_plane = self.section_plane_combo.currentData()
        low, high = self._draw_plane_section_bounds(draw_plane)
        axis_label = self._draw_plane_fixed_axis_label(draw_plane)
        self.section_top_label.setText(f"上界 {axis_label}")
        self.section_bottom_label.setText(f"下界 {axis_label}")
        self.section_top_spin.setValue(float(high))
        self.section_bottom_spin.setValue(float(low))

    def _sync_step_range_to_axis(self):
        if self._scene_object is None:
            return
        bounds = self._scene_object.bounds
        axis = self.step_axis_combo.currentData()
        if axis == "x":
            start, end = float(bounds[0]), float(bounds[1])
        elif axis == "y":
            start, end = float(bounds[2]), float(bounds[3])
        else:
            start, end = float(bounds[4]), float(bounds[5])
        span = max(abs(end - start), 1.0)
        self.step_start_spin.setValue(start)
        self.step_end_spin.setValue(end)
        self.step_size_spin.setValue(span / 10.0)

    def _emit_orthogonal_slice(self):
        if self._scene_object is None:
            return
        self.orthogonalSliceRequested.emit(
            self._scene_object.object_id,
            {
                "x": self.orthogonal_spins["x"].value(),
                "y": self.orthogonal_spins["y"].value(),
                "z": self.orthogonal_spins["z"].value(),
            },
        )

    def _emit_step_slice_batch(self):
        if self._scene_object is None:
            return
        self.axisBatchSliceRequested.emit(
            self._scene_object.object_id,
            {
                "axis": self.step_axis_combo.currentData(),
                "start": self.step_start_spin.value(),
                "end": self.step_end_spin.value(),
                "step": self.step_size_spin.value(),
            },
        )

    def _emit_plane_slice(self):
        if self._scene_object is None:
            return
        self.planeSliceRequested.emit(
            self._scene_object.object_id,
            {
                "origin": tuple(self.origin_spins[axis].value() for axis in ("x", "y", "z")),
                "normal": tuple(self.normal_spins[axis].value() for axis in ("x", "y", "z")),
            },
        )

    def _emit_start_polyline_drawing(self):
        if self._scene_object is None:
            return
        draw_plane = self.section_plane_combo.currentData()
        self.polylineDrawingRequested.emit(
            self._scene_object.object_id,
            {
                "draw_plane": draw_plane,
                "draw_value": self.section_top_spin.value(),
            },
        )

    def _emit_cancel_polyline_drawing(self):
        self.polylineDrawingCancelled.emit()

    def _emit_polyline_section(self):
        if self._scene_object is None:
            return
        draw_plane = self.section_plane_combo.currentData()
        self.polylineSectionRequested.emit(
            self._scene_object.object_id,
            {
                "top_z": self.section_top_spin.value(),
                "bottom_z": self.section_bottom_spin.value(),
                "line_step": self.section_step_spin.value(),
                "vertical_samples": self.section_vertical_spin.value(),
                "draw_plane": draw_plane,
            },
        )

    def _emit_clear(self):
        if self._scene_object is None:
            return
        self.clearDerivedRequested.emit(self._scene_object.object_id)

    def focus_slice_controls(self):
        """将焦点移动到切片模式选择区域。"""
        self.mode_checkboxes[self._current_mode].setFocus()
