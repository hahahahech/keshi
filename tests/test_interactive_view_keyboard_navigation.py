import unittest

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from gui.interactive_view import InteractiveView


class InteractiveViewKeyboardNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _new_view(self):
        return InteractiveView(
            workspace_bounds=np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=float)
        )

    def test_keyboard_navigation_step_moves_forward_with_w_key(self):
        view = self._new_view()
        try:
            camera = view.renderer.GetActiveCamera()
            camera.SetPosition(-5.0, 0.0, 0.0)
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetViewUp(0.0, 0.0, 1.0)

            handled = view.handle_navigation_key_event(
                Qt.Key.Key_W,
                pressed=True,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
            self.assertTrue(handled)
            moved = view._apply_keyboard_navigation_step(0.1)
            self.assertTrue(moved)

            position = np.array(camera.GetPosition(), dtype=float)
            focal_point = np.array(camera.GetFocalPoint(), dtype=float)
            self.assertGreater(position[0], -5.0)
            self.assertGreater(focal_point[0], 0.0)

            released = view.handle_navigation_key_event(
                Qt.Key.Key_W,
                pressed=False,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
            self.assertTrue(released)
        finally:
            view.close()

    def test_shift_boost_moves_farther_than_normal_speed(self):
        normal_view = self._new_view()
        boosted_view = self._new_view()
        try:
            normal_camera = normal_view.renderer.GetActiveCamera()
            boosted_camera = boosted_view.renderer.GetActiveCamera()
            for camera in (normal_camera, boosted_camera):
                camera.SetPosition(-5.0, 0.0, 0.0)
                camera.SetFocalPoint(0.0, 0.0, 0.0)
                camera.SetViewUp(0.0, 0.0, 1.0)

            normal_view.handle_navigation_key_event(
                Qt.Key.Key_W,
                pressed=True,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
            normal_view._apply_keyboard_navigation_step(0.1)
            normal_distance = np.linalg.norm(np.array(normal_camera.GetPosition(), dtype=float) - np.array([-5.0, 0.0, 0.0]))

            boosted_view.handle_navigation_key_event(
                Qt.Key.Key_Shift,
                pressed=True,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
            boosted_view.handle_navigation_key_event(
                Qt.Key.Key_W,
                pressed=True,
                modifiers=Qt.KeyboardModifier.NoModifier,
                auto_repeat=False,
            )
            boosted_view._apply_keyboard_navigation_step(0.1)
            boosted_distance = np.linalg.norm(np.array(boosted_camera.GetPosition(), dtype=float) - np.array([-5.0, 0.0, 0.0]))

            self.assertGreater(boosted_distance, normal_distance * 1.5)
        finally:
            normal_view.close()
            boosted_view.close()

    def test_navigation_key_ignores_ctrl_modifier(self):
        view = self._new_view()
        try:
            handled = view.handle_navigation_key_event(
                Qt.Key.Key_W,
                pressed=True,
                modifiers=Qt.KeyboardModifier.ControlModifier,
                auto_repeat=False,
            )
            self.assertFalse(handled)
            self.assertFalse(view._navigation_pressed_keys)
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
