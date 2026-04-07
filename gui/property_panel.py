"""
数据属性与派生操作控制面板。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PropertyPanel(QWidget):
    scalarChanged = pyqtSignal(str, str)
    renderModeChanged = pyqtSignal(str, str)
    colormapChanged = pyqtSignal(str, str)
    opacityChanged = pyqtSignal(str, float)
    climChanged = pyqtSignal(str, object)
    scalarBarChanged = pyqtSignal(str, bool)
    thresholdChanged = pyqtSignal(str, object)
    isosurfaceRequested = pyqtSignal(str, float)
    idwRequested = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._scene_object = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self.object_name_label = QLabel("未选择")
        self.object_kind_label = QLabel("-")
        form.addRow("名称", self.object_name_label)
        form.addRow("类型", self.object_kind_label)

        self.scalar_combo = QComboBox()
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_changed)
        form.addRow("属性", self.scalar_combo)

        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("表面", "surface")
        self.render_mode_combo.addItem("线框", "wireframe")
        self.render_mode_combo.addItem("点云", "points")
        self.render_mode_combo.addItem("体渲染", "volume")
        self.render_mode_combo.currentIndexChanged.connect(self._on_render_mode_changed)
        form.addRow("显示方式", self.render_mode_combo)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItem("Viridis（默认）", "viridis")
        self.colormap_combo.addItem("Plasma", "plasma")
        self.colormap_combo.addItem("Inferno", "inferno")
        self.colormap_combo.addItem("Turbo", "turbo")
        self.colormap_combo.addItem("冷暖", "coolwarm")
        self.colormap_combo.addItem("彩虹", "jet")
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        form.addRow("色带", self.colormap_combo)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        form.addRow("透明度", self.opacity_slider)

        self.scalar_bar_checkbox = QCheckBox("显示色标")
        self.scalar_bar_checkbox.toggled.connect(self._on_scalar_bar_toggled)
        form.addRow("", self.scalar_bar_checkbox)

        layout.addLayout(form)

        range_box = QGroupBox("属性范围")
        range_layout = QGridLayout(range_box)
        self.range_min_spin = self._make_float_spinbox()
        self.range_max_spin = self._make_float_spinbox()
        self.range_min_spin.valueChanged.connect(self._on_range_changed)
        self.range_max_spin.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(QLabel("最小值"), 0, 0)
        range_layout.addWidget(self.range_min_spin, 0, 1)
        range_layout.addWidget(QLabel("最大值"), 1, 0)
        range_layout.addWidget(self.range_max_spin, 1, 1)
        layout.addWidget(range_box)

        threshold_box = QGroupBox("阈值过滤")
        threshold_layout = QGridLayout(threshold_box)
        self.threshold_min_spin = self._make_float_spinbox()
        self.threshold_max_spin = self._make_float_spinbox()
        self.apply_threshold_button = QPushButton("应用")
        self.clear_threshold_button = QPushButton("清除")
        self.apply_threshold_button.clicked.connect(self._apply_threshold)
        self.clear_threshold_button.clicked.connect(self._clear_threshold)
        threshold_layout.addWidget(QLabel("最小值"), 0, 0)
        threshold_layout.addWidget(self.threshold_min_spin, 0, 1)
        threshold_layout.addWidget(QLabel("最大值"), 1, 0)
        threshold_layout.addWidget(self.threshold_max_spin, 1, 1)
        threshold_layout.addWidget(self.apply_threshold_button, 2, 0)
        threshold_layout.addWidget(self.clear_threshold_button, 2, 1)
        layout.addWidget(threshold_box)

        iso_box = QGroupBox("等值面")
        iso_layout = QHBoxLayout(iso_box)
        self.isovalue_spin = self._make_float_spinbox()
        self.create_isosurface_button = QPushButton("生成")
        self.create_isosurface_button.clicked.connect(self._create_isosurface)
        iso_layout.addWidget(self.isovalue_spin)
        iso_layout.addWidget(self.create_isosurface_button)
        layout.addWidget(iso_box)

        idw_box = QGroupBox("点集转规则格网")
        idw_layout = QGridLayout(idw_box)
        self.idw_nx_spin = self._make_int_spinbox(30)
        self.idw_ny_spin = self._make_int_spinbox(30)
        self.idw_nz_spin = self._make_int_spinbox(20)
        self.idw_power_spin = self._make_float_spinbox(value=2.0)
        self.idw_power_spin.setDecimals(1)
        self.idw_power_spin.setSingleStep(0.5)
        self.create_idw_button = QPushButton("生成格网")
        self.create_idw_button.clicked.connect(self._request_idw)
        idw_layout.addWidget(QLabel("NX"), 0, 0)
        idw_layout.addWidget(self.idw_nx_spin, 0, 1)
        idw_layout.addWidget(QLabel("NY"), 1, 0)
        idw_layout.addWidget(self.idw_ny_spin, 1, 1)
        idw_layout.addWidget(QLabel("NZ"), 2, 0)
        idw_layout.addWidget(self.idw_nz_spin, 2, 1)
        idw_layout.addWidget(QLabel("幂指数"), 3, 0)
        idw_layout.addWidget(self.idw_power_spin, 3, 1)
        idw_layout.addWidget(self.create_idw_button, 4, 0, 1, 2)
        layout.addWidget(idw_box)

        layout.addStretch(1)
        self._set_enabled(False)

    def _make_float_spinbox(self, value: float = 0.0):
        spin = QDoubleSpinBox(self)
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setValue(value)
        return spin

    def _make_int_spinbox(self, value: int):
        spin = QSpinBox(self)
        spin.setRange(2, 300)
        spin.setValue(value)
        return spin

    def _set_enabled(self, enabled: bool):
        for widget in (
            self.scalar_combo,
            self.render_mode_combo,
            self.colormap_combo,
            self.opacity_slider,
            self.range_min_spin,
            self.range_max_spin,
            self.scalar_bar_checkbox,
            self.threshold_min_spin,
            self.threshold_max_spin,
            self.apply_threshold_button,
            self.clear_threshold_button,
            self.isovalue_spin,
            self.create_isosurface_button,
            self.idw_nx_spin,
            self.idw_ny_spin,
            self.idw_nz_spin,
            self.idw_power_spin,
            self.create_idw_button,
        ):
            widget.setEnabled(enabled)

    def set_scene_object(self, scene_object):
        self._scene_object = scene_object
        if scene_object is None:
            self.object_name_label.setText("未选择")
            self.object_kind_label.setText("-")
            self._set_enabled(False)
            return

        dataset = scene_object.dataset
        self._updating = True
        self._set_enabled(True)
        self.object_name_label.setText(scene_object.name)
        self.object_kind_label.setText(self._dataset_kind_label(dataset.dataset_kind))
        self.scalar_combo.clear()
        self.scalar_combo.addItem("无", None)
        for scalar_name in dataset.scalar_names:
            self.scalar_combo.addItem(scalar_name, scalar_name)
        if scene_object.active_scalar:
            index = self.scalar_combo.findData(scene_object.active_scalar)
            if index >= 0:
                self.scalar_combo.setCurrentIndex(index)
        render_index = self.render_mode_combo.findData(scene_object.render_mode)
        if render_index >= 0:
            self.render_mode_combo.setCurrentIndex(render_index)
        colormap_index = self.colormap_combo.findData(scene_object.style.colormap)
        if colormap_index >= 0:
            self.colormap_combo.setCurrentIndex(colormap_index)
        self.opacity_slider.setValue(int(scene_object.opacity * 100.0))
        self.scalar_bar_checkbox.setChecked(scene_object.style.show_scalar_bar)

        scalar_range = dataset.get_scalar_range(scene_object.active_scalar) or (0.0, 1.0)
        current_clim = scene_object.style.clim or scalar_range
        self.range_min_spin.setValue(float(current_clim[0]))
        self.range_max_spin.setValue(float(current_clim[1]))
        self.threshold_min_spin.setValue(float(current_clim[0]))
        self.threshold_max_spin.setValue(float(current_clim[1]))
        self.isovalue_spin.setValue(float((current_clim[0] + current_clim[1]) / 2.0))

        is_point_set = dataset.is_point_set
        self.create_idw_button.setEnabled(is_point_set and scene_object.active_scalar is not None)
        for widget in (self.idw_nx_spin, self.idw_ny_spin, self.idw_nz_spin, self.idw_power_spin):
            widget.setEnabled(is_point_set)

        has_scalar = scene_object.active_scalar is not None
        for widget in (
            self.colormap_combo,
            self.range_min_spin,
            self.range_max_spin,
            self.scalar_bar_checkbox,
            self.threshold_min_spin,
            self.threshold_max_spin,
            self.apply_threshold_button,
            self.clear_threshold_button,
            self.isovalue_spin,
            self.create_isosurface_button,
        ):
            widget.setEnabled(has_scalar)
        self._updating = False

    def _dataset_kind_label(self, dataset_kind: str) -> str:
        mapping = {
            "regular_grid": "规则体",
            "unstructured_grid": "非规则网格",
            "surface": "曲面",
            "point_set": "点集",
            "dataset": "数据集",
        }
        return mapping.get(dataset_kind, dataset_kind)

    def _on_scalar_changed(self, _index: int):
        if self._updating or self._scene_object is None:
            return
        scalar = self.scalar_combo.currentData()
        self.scalarChanged.emit(self._scene_object.object_id, scalar or "")

    def _on_render_mode_changed(self, _index: int):
        if self._updating or self._scene_object is None:
            return
        render_mode = self.render_mode_combo.currentData()
        self.renderModeChanged.emit(self._scene_object.object_id, render_mode)

    def _on_colormap_changed(self, _index: int):
        if self._updating or self._scene_object is None:
            return
        colormap = self.colormap_combo.currentData()
        self.colormapChanged.emit(self._scene_object.object_id, colormap)

    def _on_opacity_changed(self, value: int):
        if self._updating or self._scene_object is None:
            return
        self.opacityChanged.emit(self._scene_object.object_id, value / 100.0)

    def _on_range_changed(self):
        if self._updating or self._scene_object is None:
            return
        self.climChanged.emit(
            self._scene_object.object_id,
            (self.range_min_spin.value(), self.range_max_spin.value()),
        )

    def _on_scalar_bar_toggled(self, checked: bool):
        if self._updating or self._scene_object is None:
            return
        self.scalarBarChanged.emit(self._scene_object.object_id, checked)

    def _apply_threshold(self):
        if self._scene_object is None:
            return
        self.thresholdChanged.emit(
            self._scene_object.object_id,
            (self.threshold_min_spin.value(), self.threshold_max_spin.value()),
        )

    def _clear_threshold(self):
        if self._scene_object is None:
            return
        self.thresholdChanged.emit(self._scene_object.object_id, None)

    def _create_isosurface(self):
        if self._scene_object is None:
            return
        self.isosurfaceRequested.emit(self._scene_object.object_id, self.isovalue_spin.value())

    def _request_idw(self):
        if self._scene_object is None:
            return
        self.idwRequested.emit(
            self._scene_object.object_id,
            {
                "dimensions": (
                    self.idw_nx_spin.value(),
                    self.idw_ny_spin.value(),
                    self.idw_nz_spin.value(),
                ),
                "power": self.idw_power_spin.value(),
            },
        )
