"""
地学可视化工程会话容器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.scene import SceneGraph


@dataclass
class GeologicalProject:
    name: str = "未命名"
    metadata: dict[str, Any] = field(default_factory=dict)
    scene: SceneGraph = field(default_factory=SceneGraph)
    camera_state: dict[str, Any] = field(default_factory=dict)

    def reset(self, name: str | None = None):
        self.name = name or self.name
        self.metadata = {}
        self.camera_state = {}
        self.scene = SceneGraph()
