"""
裁剪控制面板。
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
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
    clipRequested = pyqtSignal(str, object)
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

        self.clip_box = QGroupBox("框裁剪")
        clip_layout = QGridLayout(self.clip_box)
        self.clip_spins = {
            key: self._make_spinbox() for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        }
        labels = [
            ("xmin", "X 最小"),
            ("xmax", "X 最大"),
            ("ymin", "Y 最小"),
            ("ymax", "Y 最大"),
            ("zmin", "Z 最小"),
            ("zmax", "Z 最大"),
        ]
        for row, (key, label) in enumerate(labels):
            clip_layout.addWidget(QLabel(label), row, 0)
            clip_layout.addWidget(self.clip_spins[key], row, 1)
        self.create_clip_button = QPushButton("应用裁剪")
        self.create_clip_button.clicked.connect(self._emit_clip)
        clip_layout.addWidget(self.create_clip_button, 6, 0, 1, 2)
        layout.addWidget(self.clip_box)

        actions_layout = QHBoxLayout()
        self.reset_controls_button = QPushButton("重置参数")
        self.clear_derived_button = QPushButton("清除派生裁剪")
        self.reset_controls_button.clicked.connect(self._reset_from_bounds)
        self.clear_derived_button.clicked.connect(self._emit_clear)
        actions_layout.addWidget(self.reset_controls_button)
        actions_layout.addWidget(self.clear_derived_button)
        layout.addLayout(actions_layout)

        layout.addStretch(1)
        self._set_enabled(False)

    def _make_spinbox(self, default: float = 0.0):
        spin = QDoubleSpinBox(self)
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setValue(default)
        return spin

    def _set_enabled(self, enabled: bool):
        for spin in self.clip_spins.values():
            spin.setEnabled(enabled)
        for button in (self.create_clip_button, self.reset_controls_button, self.clear_derived_button):
            button.setEnabled(enabled)

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
        self.clip_spins["xmin"].setValue(float(bounds[0]))
        self.clip_spins["xmax"].setValue(float(bounds[1]))
        self.clip_spins["ymin"].setValue(float(bounds[2]))
        self.clip_spins["ymax"].setValue(float(bounds[3]))
        self.clip_spins["zmin"].setValue(float(bounds[4]))
        self.clip_spins["zmax"].setValue(float(bounds[5]))

    def _emit_clip(self):
        if self._scene_object is None:
            return
        self.clipRequested.emit(
            self._scene_object.object_id,
            (
                self.clip_spins["xmin"].value(),
                self.clip_spins["xmax"].value(),
                self.clip_spins["ymin"].value(),
                self.clip_spins["ymax"].value(),
                self.clip_spins["zmin"].value(),
                self.clip_spins["zmax"].value(),
            ),
        )

    def _emit_clear(self):
        if self._scene_object is None:
            return
        self.clearDerivedRequested.emit(self._scene_object.object_id)

    def focus_clip_controls(self):
        """将焦点移动到裁剪控制区域。"""
        self.clip_box.setFocus()
        self.create_clip_button.setFocus()
