# 地下模型高对比样例

生成文件：
- `geologic_contrast_model.csv`
- `geologic_contrast_model.vtr`

数据特点：
- 分层地层背景，适合三向切片和体渲染
- 一条断裂带，带有明显错断和破碎带
- 一个高密度高极化矿体
- 一个低密度蚀变带
- 一条高阻高磁岩脉和一个深部高密度岩塞
- 一个离散的 `lithology_index` 字段，颜色差异会比纯连续场更明显

推荐先试这些属性：
- `lithology_index`：最像地下分层模型
- `display_contrast_index`：颜色变化最强，适合快速展示
- `density_true`：看高密度矿体和低密度蚀变带
- `resistivity_log10_true`：看断裂带和岩脉
- `chargeability_mrad`：看矿体和蚀变响应