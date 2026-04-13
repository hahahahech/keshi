import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget

from gui.main_window import MainWindow


class MainWindowDockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_slice_and_clip_tools_are_dock_widgets(self):
        window = MainWindow()
        try:
            self.assertIsInstance(window.slice_window, QDockWidget)
            self.assertIsInstance(window.clip_window, QDockWidget)
            self.assertTrue(
                window.slice_window.features()
                & QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            self.assertTrue(
                window.clip_window.features()
                & QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            self.assertNotEqual(
                window.slice_window.allowedAreas() & Qt.DockWidgetArea.RightDockWidgetArea,
                Qt.DockWidgetArea.NoDockWidgetArea,
            )
            self.assertNotEqual(
                window.clip_window.allowedAreas() & Qt.DockWidgetArea.RightDockWidgetArea,
                Qt.DockWidgetArea.NoDockWidgetArea,
            )
            self.assertIn("section", window.slice_panel.mode_checkboxes)
        finally:
            window.close()

    def test_start_polyline_drawing_switches_to_top_view_and_uses_object_bounds(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            scene_object = window.scene_service.add_dataset(dataset, render=False)

            with patch.object(window.plotter, "set_view") as mock_set_view, patch.object(
                window.plotter,
                "start_polyline_drawing",
            ) as mock_start_polyline:
                window._start_polyline_drawing(scene_object.object_id, {"draw_z": 0.0})

            mock_set_view.assert_called_once_with("top")
            mock_start_polyline.assert_called_once()
            args, kwargs = mock_start_polyline.call_args
            self.assertEqual(args[0], 0.0)
            np.testing.assert_allclose(
                np.asarray(kwargs["clip_bounds"], dtype=float),
                np.asarray(scene_object.bounds, dtype=float),
            )
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
