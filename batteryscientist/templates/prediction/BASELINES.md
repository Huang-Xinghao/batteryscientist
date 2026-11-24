# 可直接使用的对比模型

## SOH 历史序列 → 多步未来 SOH

下列模型全都使用 `adapter=soh_sequence`。输入 `[batch, context]`，输出
`[batch, horizon]`，不接收未来目标。同一配置只改 `model` 即可使用相同数据和评价流程。

| model | 实现 | 训练参数 | 用途与限制 |
| --- | --- | --- | --- |
| SOHPersistence | 最后一个观测 SOH 重复 horizon 次 | 无 | 检查预测是否超过保持不变的水平 |
| SOHTrend | 在完整历史窗口拟合 OLS 直线，外推未来 | 无 | 趋势基线；context 至少2，不裁剪预测或强制单调 |
| SOHLinear | context → horizon 的全连接层 | 有 | 多步线性预测，包含偏置 |
| SOHMLP | context → hidden_dim → horizon，GELU | 有 | 小型非线性基线 |
| SOHTCN | 3个因果残差块，膨胀率1/2/4，每块两层kernel=3卷积；末位置接多步线性头 | 有 | 小型时序卷积基线，hidden_dim控制通道数；感受野29步 |

这些是模板自带的透明参考实现，不是特定论文完整复现；TCN 不应报告为某篇论文的原版结果。
如果研究涉及更强时序模型，可接入真实上游实现并记录来源、版本、适配和有效配置。
不要用这里的小型基线支持未经验证的 SOTA 声明。

```bash
python run.py --config configs/soh_history.json --model SOHTCN --hidden_dim 64 \
  --root_path /data/BatteryLife_Processed --split_file cell_splits.json \
  --protocol_file protocols/HUST_L64_H24.json --stage train --output_dir results/tcn_seed17 --seed 17
python run.py --checkpoint results/tcn_seed17 --stage test --output_dir results/tcn_seed17/test
```

用同一 protocol_file 对其他基线执行对应命令。更改模型和输出目录，有参数模型按同一规则
使用多个 seed 和验证集选参。无参数模型也沿用 stage train 保存状态，再由 stage test 恢复；
`fit_status.json` 明确标记无训练参数，不把它们当成完成了若干轮学习。

## 其他输入协议

- IC2ML：`cell_curves` 的容量曲线基线，支持原有电芯任务；不能替代只看历史 SOH 的公平比较。
- CompositionMLP：`electrolyte` 的配方特征 MLP 基线，不是 Bamboo-mixer 架构。

新增 baseline 应复用对应 adapter、划分、预处理及指标。若输入信息或任务不同，明确另设实验。
