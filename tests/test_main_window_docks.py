import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMenu

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
            self.assertFalse(hasattr(window.clip_panel, "create_clip_button"))
            self.assertFalse(hasattr(window.clip_panel, "start_grid_pick_button"))
            self.assertFalse(hasattr(window.clip_panel, "cancel_grid_pick_button"))
            self.assertTrue(hasattr(window.clip_panel, "start_mask_button"))
            self.assertTrue(hasattr(window.clip_panel, "apply_mask_button"))
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

    def test_scalar_bar_owner_follows_current_selection(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            first = window.scene_service.add_dataset(dataset, render=True, name="first")
            second = window.scene_service.add_dataset(dataset, render=True, name="second")

            window.on_object_selected(first.object_id)
            self.assertEqual(window.scene_service.render_manager._scalar_bar_owner_id, first.object_id)

            window.on_object_selected(second.object_id)
            self.assertEqual(window.scene_service.render_manager._scalar_bar_owner_id, second.object_id)
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

    def test_request_slice_move_dispatches_worker_for_slice_object(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = window.scene_service.add_dataset(dataset, render=False)
            bounds = source.bounds
            slice_object = window.scene_service.create_axis_slice(
                source.object_id,
                "z",
                (bounds[4] + bounds[5]) / 2.0,
                render=False,
                add_to_scene=True,
            )

            with patch("gui.main_window.QInputDialog.getDouble", return_value=(120.0, True)), patch.object(
                window, "_run_worker"
            ) as mock_run_worker:
                window._request_slice_move(slice_object.object_id)

            self.assertEqual(mock_run_worker.call_count, 1)
        finally:
            window.close()

    def test_request_slice_tilt_dispatches_worker_for_slice_object(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = window.scene_service.add_dataset(dataset, render=False)
            bounds = source.bounds
            slice_object = window.scene_service.create_axis_slice(
                source.object_id,
                "z",
                (bounds[4] + bounds[5]) / 2.0,
                render=False,
                add_to_scene=True,
            )

            with patch("gui.main_window.QInputDialog.getItem", return_value=("X", True)), patch(
                "gui.main_window.QInputDialog.getDouble",
                return_value=(15.0, True),
            ), patch.object(window, "_run_worker") as mock_run_worker:
                window._request_slice_tilt(slice_object.object_id)

            self.assertEqual(mock_run_worker.call_count, 1)
        finally:
            window.close()

    def test_slice_item_context_menu_contains_move_and_tilt_actions(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = window.scene_service.add_dataset(dataset, render=False)
            bounds = source.bounds
            slice_object = window.scene_service.create_axis_slice(
                source.object_id,
                "z",
                (bounds[4] + bounds[5]) / 2.0,
                render=False,
                add_to_scene=True,
            )
            item = window.scene_manager.add_object(slice_object)

            collected_actions = []

            def fake_exec(menu_self, *args, **kwargs):
                collected_actions.extend(action.text() for action in menu_self.actions())
                return None

            with patch.object(QMenu, "exec", fake_exec):
                window.scene_manager._show_item_context_menu(item, window.mapToGlobal(window.pos()))

            self.assertIn("平移切片", collected_actions)
            self.assertIn("倾斜切片", collected_actions)
        finally:
            window.close()

    def test_polyline_section_object_is_grouped_under_slice_root(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = window.scene_service.add_dataset(dataset, render=False)
            bounds = source.bounds
            section = window.scene_service.create_polyline_section(
                source.object_id,
                [
                    (bounds[0], bounds[2], bounds[5]),
                    ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, bounds[5]),
                    (bounds[1], bounds[3], bounds[5]),
                ],
                top_z=bounds[5],
                bottom_z=bounds[4],
                line_step=max((bounds[1] - bounds[0]) / 20.0, 1.0),
                vertical_samples=12,
                render=False,
                add_to_scene=False,
            )

            item = window.scene_manager.add_object(section)
            self.assertIs(item.parent(), window.scene_manager.root_nodes["slice"])
        finally:
            window.close()

    def test_create_polyline_section_dispatches_polyline_fence_slice(self):
        window = MainWindow()
        try:
            dataset = window.import_service.load_dataset("sample_data/synthetic_inversion_grid.csv")
            source = window.scene_service.add_dataset(dataset, render=False)
            points = [(0.0, 0.0, 0.0), (100.0, 50.0, 0.0)]

            with patch.object(window.plotter, "get_polyline_points", return_value=points), patch.object(
                window.scene_service,
                "create_polyline_section",
                return_value=None,
            ) as mock_polyline_section, patch.object(window, "_run_worker") as mock_run_worker:
                window._create_polyline_section(
                    source.object_id,
                    {"top_z": 0.0, "bottom_z": -200.0, "line_step": 20.0, "vertical_samples": 16},
                )
                self.assertEqual(mock_run_worker.call_count, 1)
                _, worker_func, _ = mock_run_worker.call_args[0]
                worker_func()
                mock_polyline_section.assert_called_once_with(
                    source.object_id,
                    points,
                    top_z=0.0,
                    bottom_z=-200.0,
                    line_step=20.0,
                    vertical_samples=16,
                    render=False,
                    add_to_scene=False,
                )

        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
