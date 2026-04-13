# 三维正反演测试数据

这个目录提供了 3 份可直接用于当前软件导入测试的示例数据。

## 文件说明

- `synthetic_inversion_grid.csv`
  - 规则体文本数据，包含 `x/y/z` 以及多种标量场。
  - 适合测试规则网格识别、体渲染、三向切片、步长切片、平面切片、等值面。

- `synthetic_inversion_grid.vtr`
  - 与上面同内容的 VTK `RectilinearGrid` 版本。
  - 适合测试 VTK 系列导入。

- `synthetic_forward_points.xyz`
  - 地表观测点集，包含重力和磁异常的观测值、预测值、残差。
  - 适合测试点集导入、属性切换、点集转规则体插值。

## 主要标量字段

- `density_true`
- `density_inverted`
- `density_residual`
- `resistivity_log10_true`
- `resistivity_log10_inverted`
- `chargeability_mrad`
- `magnetic_susceptibility`
- `gravity_proxy_mgal`
- `magnetic_proxy_nt`

点集文件额外包含：

- `gravity_obs_mgal`
- `gravity_pred_mgal`
- `gravity_residual_mgal`
- `mag_obs_nt`
- `mag_pred_nt`
- `mag_residual_nt`

## 建议测试顺序

1. 先导入 `synthetic_inversion_grid.csv`，检查是否识别为规则体。
2. 切换 `density_true`、`density_inverted`、`density_residual` 做体渲染和切片。
3. 再导入 `synthetic_inversion_grid.vtr`，确认 VTK 系列导入正常。
4. 最后导入 `synthetic_forward_points.xyz`，测试点集显示和点集插值成规则体。
