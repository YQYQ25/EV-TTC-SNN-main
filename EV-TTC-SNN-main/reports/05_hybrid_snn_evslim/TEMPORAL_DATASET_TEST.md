# 连续时序窗口 Dataset 测试报告

- 总体结果：通过
- 默认设置：`T=3`、`stride=3`、`batch_size=4`、`pin_memory=True`。
- 本测试只读取数据，不执行网络前向、loss、反向传播或参数更新。
- 执行命令：`EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tests/test_ttc_temporal_dataset.py`。

## 连续区间与窗口

| Split | H5样本 | 最大连续区间 | 断点 | stride=3窗口 | stride=1窗口 |
|---|---:|---:|---:|---:|---:|
| train | 27543 | 20 | 19 | 9173 | 27503 |
| val | 5165 | 4 | 3 | 1721 | 5157 |

构造规则：先按 `source_index + 1`、正向 `exp_time`、`7000±500 us` 和同序列切成最大连续区间，再在每个区间内部按 stride 生成窗口；标签和 mask 只读取窗口最后一行。

## 单元测试

- [通过] stride=3窗口起点：[0, 4, 9]
- [通过] stride=1窗口起点：[0, 1, 4, 5, 6, 9, 10, 11]
- [通过] 合成单样本shape：exp=(3, 6, 8, 8), ttc=(1, 8, 8), mask=(1, 8, 8)
- [通过] 合成窗口严格连续：source=[10, 11, 12], time=[70000.0, 77000.0, 84000.0]
- [通过] 合成标签取最后一步：window [4,5,6] 对应 H5 第6行标签
- [通过] Block增强参数一致：T步输入、最后一步TTC与mask均执行同一次90度旋转
- [通过] 合成batch shape：exp=(4, 3, 6, 8, 8), ttc=(4, 1, 8, 8)
- [通过] train随机1000窗口严格连续：windows=27503, breaks=19
- [通过] train窗口不跨断点：break_rows=[328, 913, 4984, 6111, 8397, 9725, 9739, 11324, 13580, 14110, 14538, 17381, 18181, 22581, 22868, 23055, 25212, 26669, 26684]
- [通过] train标签对应窗口最后一步：end_row=13769
- [通过] train真实单样本shape：exp=(3, 6, 360, 360), ttc=(1, 360, 360), mask=(1, 360, 360)
- [通过] val随机1000窗口严格连续：windows=5157, breaks=3
- [通过] val窗口不跨断点：break_rows=[291, 349, 1306]
- [通过] val标签对应窗口最后一步：end_row=2584
- [通过] val真实单样本shape：exp=(3, 6, 360, 360), ttc=(1, 360, 360), mask=(1, 360, 360)
- [通过] train workers=0烟雾测试：batches=20, finite=True, continuous=True, memory_stable=True
- [通过] train workers=2烟雾测试：batches=20, finite=True, continuous=True, memory_stable=True
- [通过] val workers=2烟雾测试：batches=10, finite=True, continuous=True, memory_stable=True
- [通过] val workers=4烟雾测试：batches=4, finite=True, continuous=True, memory_stable=True

## DataLoader 烟雾测试

| Split | workers | Batch数 | 首个输入shape | 总耗时(s) | 平均读取(s/batch) | mask有效像素范围 | 连续/有限 | RSS稳定增长(MiB) |
|---|---:|---:|---|---:|---:|---|---|---:|
| train | 0 | 20 | `[4, 3, 6, 360, 360]` | 1.667 | 0.076 | 39525–73891 | 是 | -7.9 |
| train | 2 | 20 | `[4, 3, 6, 360, 360]` | 0.972 | 0.040 | 40736–91178 | 是 | 13.9 |
| val | 2 | 10 | `[4, 3, 6, 360, 360]` | 0.483 | 0.039 | 41425–50604 | 是 | 0.1 |
| val | 4 | 4 | `[4, 3, 6, 360, 360]` | 0.222 | 0.039 | 41425–47007 | 是 | 15.7 |

## 结论

- `num_workers=0/2`：均正常。
- `num_workers=4`：正常。
- 是否发现窗口跨断点：否。
- 是否发现标签或 mask 错位：否。
- 实际 batch：`exp_filts=[4, 3, 6, 360, 360]`，`ttc=[4, 1, 360, 360]`，`mask=[4, 1, 360, 360]`。
- 该 Dataset 可直接向 Hybrid SNN-EV-Slim 提供严格连续的 `[B,T,6,H,W]` 输入；本阶段未启动正式训练。
