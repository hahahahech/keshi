"""
鼠标和键盘事件处理
"""

from PyQt6.QtCore import Qt

from .camera import CameraController


class EventHandler:
    """纯可视化模式下的交互事件处理器。"""

    @staticmethod
    def mouse_press_event(view, event):
        view._last_mouse_pos = event.pos()
        view.setFocus()

        button = event.button()
        modifiers = event.modifiers()

        if view.is_polyline_drawing():
            if button == Qt.MouseButton.LeftButton and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
                view.handle_polyline_click(event.pos())
                return
            if button == Qt.MouseButton.RightButton:
                view.pop_polyline_point()
                return

        if button == Qt.MouseButton.LeftButton and modifiers & Qt.KeyboardModifier.ShiftModifier:
            view._is_panning = True
            view.setCursor(Qt.CursorShape.SizeAllCursor)
        elif button == Qt.MouseButton.LeftButton:
            view._is_rotating = True
            view.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif button == Qt.MouseButton.MiddleButton:
            view._is_panning = True
            view.setCursor(Qt.CursorShape.SizeAllCursor)
        elif button == Qt.MouseButton.RightButton:
            view._is_zooming = True
            view.setCursor(Qt.CursorShape.SizeVerCursor)

    @staticmethod
    def mouse_move_event(view, event):
        if view._last_mouse_pos is None:
            view._last_mouse_pos = event.pos()
            return

        current_pos = event.pos()
        delta = current_pos - view._last_mouse_pos

        if view._is_rotating:
            CameraController.handle_rotation(view, delta)
        elif view._is_panning:
            CameraController.handle_pan(view, delta)
        elif view._is_zooming:
            CameraController.handle_zoom_drag(view, delta)
        elif view.is_polyline_drawing():
            view.handle_polyline_hover(current_pos)

        view._last_mouse_pos = current_pos
        view.view_changed.emit()

    @staticmethod
    def mouse_release_event(view, event):
        view._is_rotating = False
        view._is_panning = False
        view._is_zooming = False
        view.setCursor(Qt.CursorShape.ArrowCursor)
        view._last_mouse_pos = None

    @staticmethod
    def wheel_event(view, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.0 + (delta / 1200.0)
        CameraController.handle_zoom_wheel(view, zoom_factor)
        view.view_changed.emit()

    @staticmethod
    def key_press_event(view, event):
        if view.is_polyline_drawing():
            key = event.key()
            if key == Qt.Key.Key_Escape:
                view.cancel_polyline_drawing()
                event.accept()
                return
            if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                view.pop_polyline_point()
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                try:
                    view.finish_polyline_drawing()
                except Exception:
                    pass
                event.accept()
                return

        from pyvistaqt import QtInteractor

        QtInteractor.keyPressEvent(view, event)

    @staticmethod
    def context_menu_event(view, event):
        event.ignore()
