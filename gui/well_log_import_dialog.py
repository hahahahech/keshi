from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


NONE_COLUMN = "(无)"


class WellLogImportDialog(QDialog):
    def __init__(self, import_service, parent=None):
        super().__init__(parent)
        self.import_service = import_service
        self._schema: dict | None = None
        self._import_payload: dict | None = None
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("导入测井数据")
        self.resize(620, 520)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit(self)
        self.file_edit.setPlaceholderText("请选择测井 CSV/XYZ/DAT 文件")
        browse_button = QPushButton("浏览...", self)
        browse_button.clicked.connect(self._browse_file)
        file_row.addWidget(QLabel("数据文件", self))
        file_row.addWidget(self.file_edit, stretch=1)
        file_row.addWidget(browse_button)
        root_layout.addLayout(file_row)

        form_wrap = QWidget(self)
        form = QFormLayout(form_wrap)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.well_id_combo = QComboBox(self)
        self.x_combo = QComboBox(self)
        self.y_combo = QComboBox(self)
        self.z_combo = QComboBox(self)
        self.depth_combo = QComboBox(self)

        form.addRow("井号列", self.well_id_combo)
        form.addRow("X 列", self.x_combo)
        form.addRow("Y 列", self.y_combo)
        form.addRow("Z 列", self.z_combo)
        form.addRow("深度列", self.depth_combo)

        self.curve_list = QListWidget(self)
        self.curve_list.setMinimumHeight(130)
        form.addRow("曲线列", self.curve_list)

        self.depth_positive_down_checkbox = QCheckBox("深度向下为正（Z = 井口高程 - 深度）", self)
        self.depth_positive_down_checkbox.setChecked(True)
        form.addRow("", self.depth_positive_down_checkbox)

        self.z_reference_spin = QDoubleSpinBox(self)
        self.z_reference_spin.setRange(-1e9, 1e9)
        self.z_reference_spin.setDecimals(3)
        self.z_reference_spin.setValue(0.0)
        form.addRow("井口高程 Z", self.z_reference_spin)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("留空则使用文件名")
        form.addRow("对象名称", self.name_edit)

        root_layout.addWidget(form_wrap, stretch=1)

        self.hint_label = QLabel("提示：至少需要映射 X/Y，且 Z 与深度至少选择一个。", self)
        self.hint_label.setWordWrap(True)
        root_layout.addWidget(self.hint_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择测井数据文件",
            "",
            "文本文件 (*.csv *.xyz *.dat *.txt);;所有文件 (*)",
        )
        if not file_path:
            return
        self.file_edit.setText(file_path)
        self._load_schema(file_path)

    def _load_schema(self, file_path: str):
        try:
            self._schema = self.import_service.inspect_text_schema(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            self._schema = None
            return
        headers = list(self._schema.get("headers", []))
        self._fill_column_combos(headers)
        self._fill_curve_list(headers)
        self._apply_default_mapping(headers)
        if not self.name_edit.text().strip():
            self.name_edit.setText(Path(file_path).stem)

    def _fill_column_combos(self, headers: list[str]):
        combos = (self.well_id_combo, self.x_combo, self.y_combo, self.z_combo, self.depth_combo)
        for combo in combos:
            combo.clear()
        self.well_id_combo.addItem(NONE_COLUMN)
        self.z_combo.addItem(NONE_COLUMN)
        self.depth_combo.addItem(NONE_COLUMN)
        for header in headers:
            self.well_id_combo.addItem(header)
            self.x_combo.addItem(header)
            self.y_combo.addItem(header)
            self.z_combo.addItem(header)
            self.depth_combo.addItem(header)

    def _fill_curve_list(self, headers: list[str]):
        self.curve_list.clear()
        for header in headers:
            item = QListWidgetItem(header)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.curve_list.addItem(item)

    def _apply_default_mapping(self, headers: list[str]):
        lowered = {header.lower(): header for header in headers}
        x_name = self._pick_first(lowered, ("x", "lon", "east", "easting"), fallback_index=0, headers=headers)
        y_name = self._pick_first(lowered, ("y", "lat", "north", "northing"), fallback_index=1, headers=headers)
        z_name = self._pick_first(lowered, ("z", "elev", "elevation"), fallback_index=None, headers=headers)
        depth_name = self._pick_first(lowered, ("depth", "md", "tvd"), fallback_index=None, headers=headers)
        well_name = self._pick_first(lowered, ("well", "well_id", "hole", "borehole"), fallback_index=None, headers=headers)
        curve_name = self._pick_first(lowered, ("rt", "resistivity", "gr", "den", "ac"), fallback_index=None, headers=headers)

        self._set_combo_value(self.well_id_combo, well_name or NONE_COLUMN)
        self._set_combo_value(self.x_combo, x_name)
        self._set_combo_value(self.y_combo, y_name)
        self._set_combo_value(self.z_combo, z_name or NONE_COLUMN)
        self._set_combo_value(self.depth_combo, depth_name or NONE_COLUMN)

        if curve_name:
            for index in range(self.curve_list.count()):
                item = self.curve_list.item(index)
                if item.text() == curve_name:
                    item.setCheckState(Qt.CheckState.Checked)
                    break

    def _pick_first(
        self,
        lowered: dict[str, str],
        aliases: tuple[str, ...],
        *,
        fallback_index: int | None,
        headers: list[str],
    ) -> str | None:
        for alias in aliases:
            if alias in lowered:
                return lowered[alias]
        if fallback_index is None:
            return None
        if 0 <= fallback_index < len(headers):
            return headers[fallback_index]
        return None

    def _set_combo_value(self, combo: QComboBox, value: str):
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _selected_curve_columns(self) -> list[str]:
        selected: list[str] = []
        for index in range(self.curve_list.count()):
            item = self.curve_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def _combo_value_or_none(self, combo: QComboBox) -> str | None:
        text = combo.currentText().strip()
        if not text or text == NONE_COLUMN:
            return None
        return text

    def _build_import_payload(self) -> dict:
        file_path = self.file_edit.text().strip()
        if not file_path:
            raise ValueError("请先选择测井数据文件。")
        if self._schema is None:
            self._load_schema(file_path)
            if self._schema is None:
                raise ValueError("无法读取文件列信息。")

        x_column = self._combo_value_or_none(self.x_combo)
        y_column = self._combo_value_or_none(self.y_combo)
        z_column = self._combo_value_or_none(self.z_combo)
        depth_column = self._combo_value_or_none(self.depth_combo)
        if not x_column or not y_column:
            raise ValueError("请映射 X/Y 列。")
        if z_column is None and depth_column is None:
            raise ValueError("请至少选择 Z 列或深度列。")

        curve_columns = self._selected_curve_columns()
        spec = {
            "delimiter": self._schema.get("delimiter"),
            "has_header": self._schema.get("has_header", True),
            "well_id_column": self._combo_value_or_none(self.well_id_combo),
            "x_column": x_column,
            "y_column": y_column,
            "z_column": z_column,
            "depth_column": depth_column,
            "curve_columns": curve_columns,
            "active_scalar": curve_columns[0] if curve_columns else "depth",
            "depth_positive_down": self.depth_positive_down_checkbox.isChecked(),
            "z_reference": float(self.z_reference_spin.value()),
            "name": self.name_edit.text().strip() or Path(file_path).stem,
        }
        return {
            "file_path": file_path,
            "import_spec": spec,
        }

    def accept(self):
        try:
            self._import_payload = self._build_import_payload()
        except Exception as exc:
            QMessageBox.warning(self, "导入参数无效", str(exc))
            return
        super().accept()

    def get_import_payload(self) -> dict | None:
        return self._import_payload
