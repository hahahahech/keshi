import tempfile
import unittest
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


class MainWindowProjectIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_project_round_trip_restores_scene_and_view_state(self):
        first_window = MainWindow()
        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.close()
        file_path = handle.name
        try:
            dataset = first_window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = first_window.scene_service.add_dataset(dataset, render=False)
            first_window.scene_manager.add_object(source)
            first_window.scene_service.update_style(
                source.object_id,
                render_mode="wireframe",
                opacity=0.3,
                colormap="plasma",
            )
            bounds = source.bounds
            slice_object = first_window.scene_service.create_axis_slice(
                source.object_id,
                "z",
                (bounds[4] + bounds[5]) / 2.0,
                render=False,
                add_to_scene=True,
            )
            first_window.scene_manager.add_object(slice_object)

            custom_bounds = np.array([-50.0, 950.0, -80.0, 1080.0, -600.0, 120.0], dtype=float)
            first_window.plotter.set_workspace_bounds(custom_bounds)
            first_window.plotter.set_projection_mode(True)
            first_window.plotter.set_view("left")
            saved_camera = first_window.plotter.get_camera_info()

            first_window.save_project_to_path(file_path)

            second_window = MainWindow()
            try:
                second_window.open_project_from_path(file_path)

                restored_objects = second_window.scene_service.all_objects()
                self.assertEqual(len(restored_objects), 2)
                restored_dataset = next(
                    scene_object
                    for scene_object in restored_objects
                    if scene_object.object_type == "dataset"
                )
                self.assertEqual(restored_dataset.render_mode, "wireframe")
                self.assertAlmostEqual(restored_dataset.opacity, 0.3, places=6)
                self.assertEqual(restored_dataset.style.colormap, "plasma")
                self.assertTrue(second_window.plotter.get_projection_mode())
                np.testing.assert_allclose(
                    second_window.plotter.get_workspace_bounds(),
                    custom_bounds,
                )
                np.testing.assert_allclose(
                    np.asarray(second_window.plotter.get_camera_info()["position"], dtype=float),
                    np.asarray(saved_camera["position"], dtype=float),
                )
            finally:
                second_window.close()
        finally:
            first_window.close()
            Path(file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
