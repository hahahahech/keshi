"""
Project container for geological visualization sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.scene import SceneGraph


@dataclass
class GeologicalProject:
    name: str = "untitled"
    metadata: dict[str, Any] = field(default_factory=dict)
    scene: SceneGraph = field(default_factory=SceneGraph)
