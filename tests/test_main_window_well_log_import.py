import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication, QDialog

from gui.main_window import MainWindow


class _FakeWellDialog:
    def __init__(self, payload):
        self._payload = payload

    def exec(self):
        return int(QDialog.DialogCode.Accepted)

    def get_import_payload(self):
        return self._payload


class MainWindowWellLogImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _write_temp_csv(self, content: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        try:
            handle.write(content)
            handle.flush()
            return handle.name
        finally:
            handle.close()

    def test_import_well_log_data_adds_drillhole_object_to_scene(self):
        window = MainWindow()
        path = self._write_temp_csv(
            "\n".join(
                [
                    "well_id,x,y,depth,rt",
                    "A,100,200,0,10",
                    "A,100,200,10,15",
                    "A,100,200,20,20",
                ]
            )
        )
        payload = {
            "file_path": path,
            "import_spec": {
                "well_id_column": "well_id",
                "x_column": "x",
                "y_column": "y",
                "depth_column": "depth",
                "curve_columns": ["rt"],
                "depth_positive_down": True,
                "z_reference": 0.0,
                "name": "井A",
            },
        }
        try:
            fake_dialog = _FakeWellDialog(payload)
            with patch("gui.main_window.WellLogImportDialog", return_value=fake_dialog), patch.object(
                window,
                "_run_worker",
                side_effect=lambda _desc, func, on_success: on_success(func()),
            ):
                window.import_well_log_data()

            drillhole_objects = [obj for obj in window.scene_service.all_objects() if obj.object_type == "drillhole"]
            self.assertGreaterEqual(len(drillhole_objects), 1)
            target = drillhole_objects[-1]
            self.assertEqual(target.name, "井A")
            self.assertGreater(target.data.n_points, 0)
            self.assertGreaterEqual(target.data.n_lines, 1)
        finally:
            Path(path).unlink(missing_ok=True)
            window.close()

    def test_well_log_import_maps_coordinates_to_reference_dataset_range(self):
        window = MainWindow()
        try:
            base_dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            base_object = window.scene_service.add_dataset(base_dataset, render=False, object_type="dataset")

            well_dataset = window.import_service.load_well_log_dataset(
                "sample_data/sample_borehole_data.csv",
                {
                    "x_column": "X",
                    "y_column": "Y",
                    "z_column": "Z",
                    "depth_column": "Depth_m",
                    "curve_columns": ["RES_ohm_m"],
                    "name": "borehole",
                },
            )

            window._on_well_log_import_finished(well_dataset)
            all_objects = window.scene_service.all_objects()
            well_objects = [scene_object for scene_object in all_objects if scene_object.name == "borehole"]
            self.assertEqual(len(well_objects), 1)
            well_bounds = tuple(float(v) for v in well_objects[0].bounds)
            base_bounds = tuple(float(v) for v in base_object.bounds)

            self.assertGreaterEqual(well_bounds[0], base_bounds[0] - 1e-6)
            self.assertLessEqual(well_bounds[1], base_bounds[1] + 1e-6)
            self.assertGreaterEqual(well_bounds[2], base_bounds[2] - 1e-6)
            self.assertLessEqual(well_bounds[3], base_bounds[3] + 1e-6)
            self.assertGreaterEqual(well_bounds[4], base_bounds[4] - 1e-6)
            self.assertLessEqual(well_bounds[5], base_bounds[5] + 1e-6)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
