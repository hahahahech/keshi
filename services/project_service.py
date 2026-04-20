"""
工程保存与加载辅助服务。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class ProjectService:
    def save_project(
        self,
        file_path: str,
        *,
        project_name: str,
        metadata: dict[str, Any],
        camera_state: dict[str, Any],
        objects: list[dict[str, Any]],
        view_state: dict[str, Any] | None = None,
    ):
        normalized_camera_state = _json_ready(camera_state or {})
        normalized_view_state = _json_ready(view_state or {})
        if normalized_camera_state and "camera_state" not in normalized_view_state:
            normalized_view_state["camera_state"] = normalized_camera_state
        payload = {
            "version": 2,
            "project_meta": {
                "name": project_name,
            },
            "name": project_name,
            "metadata": _json_ready(metadata),
            "camera_state": normalized_camera_state,
            "view_state": normalized_view_state,
            "objects": _json_ready(objects),
        }
        path = Path(file_path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_project(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        project_meta = payload.get("project_meta", {})
        normalized_name = project_meta.get("name") or payload.get("name", "未命名")
        normalized_camera_state = payload.get("camera_state") or {}
        normalized_view_state = payload.get("view_state") or {}
        if normalized_camera_state and "camera_state" not in normalized_view_state:
            normalized_view_state["camera_state"] = normalized_camera_state
        return {
            **payload,
            "project_meta": {
                **project_meta,
                "name": normalized_name,
            },
            "name": normalized_name,
            "metadata": payload.get("metadata", {}),
            "camera_state": normalized_camera_state or normalized_view_state.get("camera_state", {}),
            "view_state": normalized_view_state,
            "objects": payload.get("objects", []),
        }
