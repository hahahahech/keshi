"""
工程保存与加载辅助服务。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProjectService:
    def save_project(
        self,
        file_path: str,
        *,
        project_name: str,
        metadata: dict[str, Any],
        camera_state: dict[str, Any],
        objects: list[dict[str, Any]],
    ):
        payload = {
            "version": 1,
            "name": project_name,
            "metadata": metadata,
            "camera_state": camera_state,
            "objects": objects,
        }
        path = Path(file_path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_project(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        return json.loads(path.read_text(encoding="utf-8"))
