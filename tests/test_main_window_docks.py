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

    def test_view_menu_scalar_bar_action_tracks_selected_object(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            scene_object = window.scene_service.add_dataset(dataset, render=False)

            self.assertFalse(window.toggle_scalar_bar_action.isEnabled())

            window.on_object_selected(scene_object.object_id)

            self.assertTrue(window.toggle_scalar_bar_action.isEnabled())
            self.assertTrue(window.toggle_scalar_bar_action.isChecked())

            window.toggle_scalar_bar_action.setChecked(False)

            updated = window.scene_service.get_object(scene_object.object_id)
            self.assertFalse(updated.style.show_scalar_bar)
            self.assertFalse(window.property_panel.scalar_bar_checkbox.isChecked())
        finally:
            window.close()

    def test_view_menu_visibility_actions_track_component_state(self):
        window = MainWindow()
        try:
            self.assertTrue(window.toggle_axes_action.isChecked())
            self.assertFalse(window.toggle_axis_scales_action.isChecked())
            self.assertTrue(window.toggle_selection_highlight_action.isChecked())
            self.assertFalse(hasattr(window, "toggle_origin_axes_action"))

            window.toggle_axes_action.setChecked(False)
            window.toggle_axis_scales_action.setChecked(True)

            self.assertFalse(window.view_axes.isVisible())
            self.assertTrue(window.axis_scale_component.get_visible())
        finally:
            window.close()

    def test_view_menu_no_longer_contains_grid_or_origin_axes(self):
        window = MainWindow()
        try:
            view_menu = next(
                action.menu()
                for action in window.menuBar().actions()
                if action.menu() is not None and action.text() == "视图"
            )
            action_texts = {action.text() for action in view_menu.actions()}

            self.assertNotIn("网格", action_texts)
            self.assertNotIn("原点坐标轴", action_texts)
        finally:
            window.close()

    def test_selection_highlight_toggle_prevents_future_highlights(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            scene_object = window.scene_service.add_dataset(dataset, render=False)

            window.toggle_selection_highlight_action.setChecked(False)
            window.on_object_selected(scene_object.object_id)

            self.assertEqual(window._selection_outline_actors, [])
            self.assertFalse(window.toggle_selection_highlight_action.isChecked())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
