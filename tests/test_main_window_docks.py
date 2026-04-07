import unittest

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
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
