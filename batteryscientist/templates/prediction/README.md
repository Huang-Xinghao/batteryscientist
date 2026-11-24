# 可扩展预测实验模板

所有输入协议共用 `run.py → exp/experiment.py` 的训练、检查、验证、checkpoint 恢复和结果导出。
研究模型、数据及统计口径在对应模块实现；IC2ML 是其中一个具体实现。

先读 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) 中的数据划分、输入信息、选型与评价约定，
再按 [BASELINES.md](BASELINES.md) 选择公平对比。准备工具生成项目固定电芯名单
`cell_splits.json`，并配置共享的 `comparison_protocol.json`，后续候选模型共用这组比较条件。

## 两种电芯输入协议

| adapter | 输入 | 示例模型 | 标签及适用协议 |
| --- | --- | --- | --- |
| `cell_curves`（电芯默认） | 多周期电压/电流/容量曲线 | IC2ML | 保留 BatteryLife 原有早期窗口、寿命标签资格筛选；支持 life/soh/trajectory/rul |
| `soh_sequence` | 历史周期容量除以额定容量的 SOH 序列 | SOHLinear | 未来 SOH trajectory；不需要寿命标签或充放电曲线 |

自带持久性、OLS趋势、线性、MLP和小型TCN基线；它们不代表研究方法已实现。新增模型时，在
`models/` 实现并注册到 `models/registry.py`，将名称加入对应 adapter 的 `compatible_models`。
SOH 序列模型输入为 `[batch, context]`，输出为 `[batch, horizon]`。
模型可以定义 `validate_config(args)` 静态方法和 `config_keys` 元组；前者验证模型参数，
后者标明 checkpoint 恢复必须一致的模型配置。新参数在 `configuration.py` 声明。
IC2ML 的维度和曲线长度限制只用于 IC2ML。

## 历史 SOH 预测示例

在 `configs/soh_history.json` 基础上设置数据路径与划分，或运行：

```bash
python run.py --config configs/soh_history.json --root_path /data/BatteryLife_Processed \
  --split_file /path/to/proposed_cell_splits.json --stage check --output_dir results/check
python run.py --config configs/soh_history.json --root_path /data/BatteryLife_Processed \
  --split_file /path/to/proposed_cell_splits.json --stage train --output_dir results/train
python run.py --checkpoint results/train --stage test --output_dir results/test
```

从项目根目录运行时，配置、split_file、输出路径均相对于项目根目录；准备工具会将显式数据和划分路径转成绝对路径。

## 数据与划分

电芯 `split_file` 支持两种格式：

```json
{"train": ["HUST_1.pkl"], "val": ["HUST_2.pkl"], "test": ["HUST_3.pkl"]}
```

或 `{"datasets": {"HUST": {"train": [...], "val": [...], "test": [...]}}}`。
不提供时沿用 BatteryLife 固定名单。电解液仍使用行索引名单与配方分组。
SOH 数据按 `root/dataset/filename`、`root/filename`、BatteryLife 路径约定查找。

SOH 加载器按原始循环编号排序，逐周期取有限放电容量的最大值，除以文件中的额定容量。
没有全局拟合或归一化。无效容量所在窗口被排除。
`cycle_coordinate=record` 按保存记录建窗；`cycle_number` 只保留原始编号间隔等于
`cycle_step` 的窗口，不把缺失周期拼成连续周期。预设采用后者，64 步历史预测未来 24 步。
`eval_cycle_min/max` 限定观测终点，单位由 cycle_coordinate 决定。
源文件哈希、实际电池名单和窗口数写入 data_manifest，恢复时核对。

## 采样、损失、评价及选型

- `sampling=window`：窗口均匀；`cell_balanced`：有放回采样，每个电池的预期采样权重相同。
  自定义数据加载器可以提供自己的 sampler。
- `loss=MSE/MAE/Huber`：默认按观测元素计算损失；研究损失在 adapter 的 `loss_parts`
  覆盖，返回可微损失和、对应权重。`optimizer(model)` 可覆盖优化器。
- `aggregation=observation`：全部观测值汇总；`cell_macro`：每电池先算 MAE/RMSE，
  再电池等权平均，同时输出 per_cell。自定义评价在 adapter 的 `metrics` 覆盖。
- `selection_metric=loss`：按验证损失选 checkpoint；`trajectory.rmse` 等：按指定的
  验证指标选 checkpoint，当前均以越小越好。训练目标与选型指标独立。

SOH 预设采用电池均衡采样、电池宏平均和验证 trajectory.rmse。
每轮记录实际选型指标和值；训练后恢复最佳权重。候选训练不计算测试指标。
模型的检查/训练/恢复/导出均经过同一套入口。CSV 包含样本、任务、目标坐标及可用的电池/观测位置标识。

## 扩展位置

- 新输入或预处理：增加 `data_provider/` 加载器与 `adapters/` 实现，注册到 ADAPTER_REGISTRY。
  在 `state()` 保存训练拟合状态与划分，`loaders(state)` 恢复并核对；forward 不读取未来目标作为输入。
- 新模型：修改 `models/` 与 registry；无需复制训练循环。
- 自定义训练策略：优先覆盖 adapter 的 loss_parts/optimizer；确需修改循环时直接修改共享实现。

## 维护与检查

本目录是模板源码，压缩包是准备工具分发的构建产物。在仓库根目录运行：

```bash
python scripts/bundle_prediction_template.py batteryscientist/templates/prediction
cd batteryscientist/templates/prediction
python -m pytest tests -q
```

测试覆盖 IC2ML、电解液与 SOH 序列。它们验证实现语义，不代表模型在真实研究数据上有效。
