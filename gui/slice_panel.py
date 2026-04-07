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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class SlicePanel(QWidget):
    axisBatchSliceRequested = pyqtSignal(str, object)
    orthogonalSliceRequested = pyqtSignal(str, object)
    planeSliceRequested = pyqtSignal(str, object)
    clearDerivedRequested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene_object = None
        self._current_mode = "orthogonal"
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
        ):
            checkbox = QCheckBox(label, self)
            checkbox.toggled.connect(
                lambda checked, key=mode: self._on_mode_toggled(key, checked)
            )
            mode_layout.addWidget(checkbox)
            self.mode_checkboxes[mode] = checkbox
        layout.addWidget(mode_box)

        self.mode_stack = QStackedWidget(self)
        self.mode_stack.addWidget(self._build_orthogonal_page())
        self.mode_stack.addWidget(self._build_step_page())
        self.mode_stack.addWidget(self._build_plane_page())
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
        self.orthogonal_spins = {axis: self._make_spinbox() for axis in ("x", "y", "z")}
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
        self.step_start_spin = self._make_spinbox()
        self.step_end_spin = self._make_spinbox()
        self.step_size_spin = self._make_spinbox(default=10.0, minimum=0.000001)
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
        self.origin_spins = {axis: self._make_spinbox() for axis in ("x", "y", "z")}
        self.normal_spins = {
            axis: self._make_spinbox(default=1.0 if axis == "z" else 0.0)
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

    def _make_spinbox(
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
        ]
        widgets.extend(self.orthogonal_spins.values())
        widgets.extend(self.origin_spins.values())
        widgets.extend(self.normal_spins.values())
        for widget in widgets:
            widget.setEnabled(enabled)
        for checkbox in self.mode_checkboxes.values():
            checkbox.setEnabled(enabled)

    def _set_mode(self, mode: str):
        self._current_mode = mode
        order = {"orthogonal": 0, "step": 1, "plane": 2}
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
        if scene_object is None:
            self.target_label.setText("未选择")
            self._set_enabled(False)
            return
        self.target_label.setText(scene_object.name)
        self._set_enabled(True)
        self._reset_from_bounds()

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
        self._sync_step_range_to_axis()

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

    def _emit_clear(self):
        if self._scene_object is None:
            return
        self.clearDerivedRequested.emit(self._scene_object.object_id)

    def focus_slice_controls(self):
        """将焦点移动到切片模式选择区域。"""
        self.mode_checkboxes[self._current_mode].setFocus()
