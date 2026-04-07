"""
基于 QRunnable 的简单后台任务执行器。
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception as exc:
            message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.signals.failed.emit(message)
            return
        self.signals.finished.emit(result)
