import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication

from gui.interactive_view import InteractiveView


class InteractiveViewPolylineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_polyline_drawing_state_round_trip(self):
        view = InteractiveView()
        try:
            view.start_polyline_drawing(12.0)
            self.assertTrue(view.is_polyline_drawing())

            view.add_polyline_point((1.0, 2.0, 12.0))
            view.add_polyline_point((4.0, 6.0, 12.0))

            self.assertEqual(len(view.get_polyline_points()), 2)
            self.assertAlmostEqual(view.get_polyline_points()[0][2], 12.0)

            view.cancel_polyline_drawing()
            self.assertFalse(view.is_polyline_drawing())
            self.assertEqual(view.get_polyline_points(), [])
        finally:
            view.close()

    def test_polyline_hover_is_clamped_to_active_object_bounds(self):
        view = InteractiveView()
        try:
            bounds = np.array([0.0, 10.0, 20.0, 30.0, -50.0, 50.0], dtype=float)
            view.start_polyline_drawing(12.0, clip_bounds=bounds)
            view.add_polyline_point((5.0, 25.0, 12.0))

            with patch(
                "gui.interactive_view.view.CoordinateConverter.screen_to_horizontal_plane",
                return_value=np.array([18.0, 42.0, 12.0], dtype=float),
            ) as mock_project:
                view.handle_polyline_hover(QPoint(120, 80))

            np.testing.assert_allclose(
                view._polyline_hover_point,
                np.array([10.0, 30.0, 12.0], dtype=float),
            )
            self.assertFalse(mock_project.call_args.kwargs.get("clip_to_bounds", True))
        finally:
            view.close()

    def test_polyline_click_is_clamped_to_active_object_bounds(self):
        view = InteractiveView()
        try:
            bounds = np.array([0.0, 10.0, 20.0, 30.0, -50.0, 50.0], dtype=float)
            view.start_polyline_drawing(12.0, clip_bounds=bounds)

            with patch(
                "gui.interactive_view.view.CoordinateConverter.screen_to_horizontal_plane",
                return_value=np.array([-8.0, 36.0, 12.0], dtype=float),
            ) as mock_project:
                view.handle_polyline_click(QPoint(220, 180))

            self.assertEqual(view.get_polyline_points(), [(0.0, 30.0, 12.0)])
            self.assertFalse(mock_project.call_args.kwargs.get("clip_to_bounds", True))
        finally:
            view.close()

    def test_top_view_places_camera_above_focal_point(self):
        view = InteractiveView()
        try:
            view.set_view("top")
            camera = view.renderer.GetActiveCamera()
            position = np.array(camera.GetPosition(), dtype=float)
            focal_point = np.array(camera.GetFocalPoint(), dtype=float)
            self.assertGreater(position[2], focal_point[2])
        finally:
            view.close()

    def test_bottom_view_places_camera_below_focal_point(self):
        view = InteractiveView()
        try:
            view.set_view("bottom")
            camera = view.renderer.GetActiveCamera()
            position = np.array(camera.GetPosition(), dtype=float)
            focal_point = np.array(camera.GetFocalPoint(), dtype=float)
            self.assertLess(position[2], focal_point[2])
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
