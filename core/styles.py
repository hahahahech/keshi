"""
Render style primitives shared by scene objects.
"""

from __future__ import annotations

from dataclasses import dataclass


def _normalize_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    red, green, blue = color
    return (
        max(0.0, min(float(red), 1.0)),
        max(0.0, min(float(green), 1.0)),
        max(0.0, min(float(blue), 1.0)),
    )


@dataclass
class RenderStyle:
    color: tuple[float, float, float] = (0.75, 0.78, 0.85)
    opacity: float = 1.0
    visible: bool = True
    render_mode: str = "surface"

    def normalized(self) -> "RenderStyle":
        return RenderStyle(
            color=_normalize_color(self.color),
            opacity=max(0.0, min(float(self.opacity), 1.0)),
            visible=bool(self.visible),
            render_mode=self.render_mode,
        )
