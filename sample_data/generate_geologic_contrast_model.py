from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "geologic_contrast_model.csv"
VTR_PATH = ROOT / "geologic_contrast_model.vtr"
README_PATH = ROOT / "geologic_contrast_model_README.md"


def gaussian_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radius
    return np.exp(
        -(
            ((x - cx) / rx) ** 2
            + ((y - cy) / ry) ** 2
            + ((z - cz) / rz) ** 2
        )
    )


def smooth_field(field: np.ndarray, passes: int = 2) -> np.ndarray:
    result = field.astype(float, copy=True)
    for _ in range(max(int(passes), 0)):
        result = (
            result * 4.0
            + np.roll(result, 1, axis=0)
            + np.roll(result, -1, axis=0)
            + np.roll(result, 1, axis=1)
            + np.roll(result, -1, axis=1)
            + np.roll(result, 1, axis=2)
            + np.roll(result, -1, axis=2)
        ) / 10.0
    return result


def build_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1200.0, 25)
    y = np.linspace(0.0, 950.0, 20)
    z = np.linspace(-540.0, 0.0, 10)
    return x, y, z


def build_fields() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    x_axis, y_axis, z_axis = build_axes()
    x, y, z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    fault_trace = 590.0 + 0.14 * (y - 450.0)
    fault_offset = np.where(x > fault_trace, 85.0, 0.0)
    shifted_x = x - fault_offset

    layer_1 = -70.0 - 20.0 * np.sin(shifted_x / 150.0) - 12.0 * np.cos(y / 130.0)
    layer_2 = -190.0 - 28.0 * np.sin((shifted_x + 0.40 * y) / 190.0) + 18.0 * np.cos(y / 160.0)
    layer_3 = -360.0 - 36.0 * np.cos(shifted_x / 210.0) + 20.0 * np.sin(y / 170.0)
    layer_4 = -500.0 - 15.0 * np.sin((shifted_x - y) / 180.0)

    lithology_index = np.select(
        [z >= layer_1, z >= layer_2, z >= layer_3, z >= layer_4],
        [1, 2, 3, 4],
        default=5,
    ).astype(np.int16)

    ore_body = gaussian_3d(x, y, z, center=(790.0, 330.0, -325.0), radius=(115.0, 95.0, 85.0))
    alteration_zone = gaussian_3d(x, y, z, center=(290.0, 620.0, -160.0), radius=(170.0, 140.0, 70.0))
    basalt_plug = gaussian_3d(x, y, z, center=(990.0, 690.0, -430.0), radius=(90.0, 80.0, 150.0))
    channel_fill = gaussian_3d(x, y, z, center=(180.0, 230.0, -55.0), radius=(280.0, 90.0, 40.0))

    dyke_center = 460.0 + 0.22 * (y - 450.0)
    dyke = np.exp(-(((x - dyke_center) / 40.0) ** 2 + ((z + 280.0) / 260.0) ** 2))

    fault_damage = np.exp(-(((x - fault_trace) / 45.0) ** 2)) * np.exp(-((z + 280.0) / 260.0) ** 2)

    lithology_index = np.where(alteration_zone > 0.48, 6, lithology_index)
    lithology_index = np.where(dyke > 0.52, 7, lithology_index)
    lithology_index = np.where(ore_body > 0.42, 8, lithology_index)
    lithology_index = np.where(basalt_plug > 0.40, 9, lithology_index)
    lithology_index = np.where(channel_fill > 0.45, 10, lithology_index)

    density_base = np.array([0.0, 1.82, 2.08, 2.35, 2.68, 2.95], dtype=float)
    resistivity_base = np.array([0.0, 1.25, 1.85, 1.45, 2.20, 2.60], dtype=float)
    chargeability_base = np.array([0.0, 8.0, 12.0, 18.0, 10.0, 6.0], dtype=float)
    susceptibility_base = np.array([0.0, 0.002, 0.007, 0.012, 0.005, 0.009], dtype=float)

    density_true = density_base[lithology_index.clip(max=5)]
    density_true = (
        density_true
        + 0.70 * ore_body
        - 0.40 * alteration_zone
        + 0.30 * dyke
        + 0.22 * basalt_plug
        - 0.18 * fault_damage
        - 0.15 * channel_fill
    )
    density_true = np.clip(density_true, 1.55, 3.55)

    resistivity_log10_true = resistivity_base[lithology_index.clip(max=5)]
    resistivity_log10_true = (
        resistivity_log10_true
        + 0.55 * dyke
        - 0.42 * fault_damage
        + 0.30 * ore_body
        - 0.35 * alteration_zone
        - 0.20 * channel_fill
        + 0.25 * basalt_plug
    )
    resistivity_log10_true = np.clip(resistivity_log10_true, 0.80, 3.20)

    chargeability_mrad = chargeability_base[lithology_index.clip(max=5)]
    chargeability_mrad = (
        chargeability_mrad
        + 58.0 * ore_body
        + 20.0 * fault_damage
        + 18.0 * alteration_zone
        - 8.0 * dyke
        - 4.0 * channel_fill
    )
    chargeability_mrad = np.clip(chargeability_mrad, 2.0, 95.0)

    magnetic_susceptibility = susceptibility_base[lithology_index.clip(max=5)]
    magnetic_susceptibility = (
        magnetic_susceptibility
        + 0.065 * ore_body
        + 0.030 * dyke
        + 0.018 * basalt_plug
        - 0.004 * alteration_zone
    )
    magnetic_susceptibility = np.clip(magnetic_susceptibility, 0.0005, None)

    density_inverted = 0.92 * smooth_field(density_true, passes=2) + 0.08 * smooth_field(density_true, passes=4)
    density_inverted = density_inverted - 0.06 * ore_body + 0.03 * alteration_zone + 0.02 * fault_damage
    density_residual = density_inverted - density_true

    resistivity_log10_inverted = (
        0.90 * smooth_field(resistivity_log10_true, passes=2)
        + 0.10 * smooth_field(resistivity_log10_true, passes=4)
    )
    resistivity_log10_inverted = resistivity_log10_inverted - 0.05 * dyke + 0.04 * fault_damage

    gravity_proxy_mgal = (
        (density_true - density_true.mean()) * (10.5 - 2.5 * (np.abs(z) / np.abs(z_axis.min())))
        + 2.4 * ore_body
        - 1.4 * alteration_zone
        + 0.8 * basalt_plug
    )
    magnetic_proxy_nt = (
        magnetic_susceptibility * 950.0
        + 22.0 * dyke
        + 16.0 * ore_body
        + 10.0 * basalt_plug
        - 4.0 * alteration_zone
    )

    display_contrast_index = (
        lithology_index.astype(float)
        + 2.2 * ore_body
        + 1.4 * dyke
        + 1.0 * basalt_plug
        - 1.1 * alteration_zone
        - 0.6 * fault_damage
    )

    fields = {
        "lithology_index": lithology_index.astype(np.int16),
        "density_true": density_true,
        "density_inverted": density_inverted,
        "density_residual": density_residual,
        "resistivity_log10_true": resistivity_log10_true,
        "resistivity_log10_inverted": resistivity_log10_inverted,
        "chargeability_mrad": chargeability_mrad,
        "magnetic_susceptibility": magnetic_susceptibility,
        "gravity_proxy_mgal": gravity_proxy_mgal,
        "magnetic_proxy_nt": magnetic_proxy_nt,
        "display_contrast_index": display_contrast_index,
    }
    return (x_axis, y_axis, z_axis), fields


def save_csv(axes: tuple[np.ndarray, np.ndarray, np.ndarray], fields: dict[str, np.ndarray]) -> None:
    x_axis, y_axis, z_axis = axes
    x, y, z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    ordered_names = ["x", "y", "z", *fields.keys()]
    columns = [
        x.ravel(order="F"),
        y.ravel(order="F"),
        z.ravel(order="F"),
        *[np.asarray(values).ravel(order="F") for values in fields.values()],
    ]
    stacked = np.column_stack(columns)
    header = ",".join(ordered_names)
    fmt = [
        "%.3f",
        "%.3f",
        "%.3f",
        "%.0f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.8f",
        "%.6f",
        "%.6f",
        "%.6f",
    ]
    np.savetxt(CSV_PATH, stacked, delimiter=",", header=header, comments="", fmt=fmt)


def save_vtr(axes: tuple[np.ndarray, np.ndarray, np.ndarray], fields: dict[str, np.ndarray]) -> None:
    x_axis, y_axis, z_axis = axes
    grid = pv.RectilinearGrid(x_axis, y_axis, z_axis)
    for name, values in fields.items():
        grid.point_data[name] = np.asarray(values).ravel(order="F")
    grid.save(VTR_PATH)


def save_readme() -> None:
    README_PATH.write_text(
        "\n".join(
            [
                "# 地下模型高对比样例",
                "",
                "生成文件：",
                "- `geologic_contrast_model.csv`",
                "- `geologic_contrast_model.vtr`",
                "",
                "数据特点：",
                "- 分层地层背景，适合三向切片和体渲染",
                "- 一条断裂带，带有明显错断和破碎带",
                "- 一个高密度高极化矿体",
                "- 一个低密度蚀变带",
                "- 一条高阻高磁岩脉和一个深部高密度岩塞",
                "- 一个离散的 `lithology_index` 字段，颜色差异会比纯连续场更明显",
                "",
                "推荐先试这些属性：",
                "- `lithology_index`：最像地下分层模型",
                "- `display_contrast_index`：颜色变化最强，适合快速展示",
                "- `density_true`：看高密度矿体和低密度蚀变带",
                "- `resistivity_log10_true`：看断裂带和岩脉",
                "- `chargeability_mrad`：看矿体和蚀变响应",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    axes, fields = build_fields()
    save_csv(axes, fields)
    save_vtr(axes, fields)
    save_readme()
    point_count = int(np.prod([axis.size for axis in axes]))
    print(f"generated {CSV_PATH.name} and {VTR_PATH.name} with {point_count} points")


if __name__ == "__main__":
    main()
