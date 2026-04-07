"""
场景对象共享的渲染样式定义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _normalize_color(color: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if color is None:
        return None
    red, green, blue = color
    return (
        max(0.0, min(float(red), 1.0)),
        max(0.0, min(float(green), 1.0)),
        max(0.0, min(float(blue), 1.0)),
    )


def _normalize_range(
    range_value: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    if range_value is None:
        return None
    low, high = float(range_value[0]), float(range_value[1])
    if low > high:
        low, high = high, low
    return (low, high)


@dataclass
class RenderStyle:
    color: tuple[float, float, float] | None = (0.75, 0.78, 0.85)
    opacity: float = 1.0
    visible: bool = True
    render_mode: str = "surface"
    scalar_name: str | None = None
    colormap: str = "viridis"
    clim: tuple[float, float] | None = None
    show_scalar_bar: bool = True
    volume_preset: str = "linear"
    threshold_range: tuple[float, float] | None = None
    opacity_curve: list[float] = field(default_factory=lambda: [0.0, 0.15, 0.35, 0.75, 1.0])

    def normalized(self) -> "RenderStyle":
        return RenderStyle(
            color=_normalize_color(self.color),
            opacity=max(0.0, min(float(self.opacity), 1.0)),
            visible=bool(self.visible),
            render_mode=self.render_mode,
            scalar_name=self.scalar_name,
            colormap=self.colormap or "viridis",
            clim=_normalize_range(self.clim),
            show_scalar_bar=bool(self.show_scalar_bar),
            volume_preset=self.volume_preset or "linear",
            threshold_range=_normalize_range(self.threshold_range),
            opacity_curve=[max(0.0, min(float(value), 1.0)) for value in self.opacity_curve],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.normalized())
        if payload["clim"] is not None:
            payload["clim"] = list(payload["clim"])
        if payload["threshold_range"] is not None:
            payload["threshold_range"] = list(payload["threshold_range"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RenderStyle":
        if not payload:
            return cls().normalized()
        return cls(**payload).normalized()
