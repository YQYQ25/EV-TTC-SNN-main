# M3ED 1k 事件分片与 TTC 标签调试集实现指令

根据 `总体思路的报告文件.md`，开始第一阶段实现，但暂时不要接入网络训练。

## 目标

选取一个小规模 M3ED 训练序列（spot_outdoor_day_skatepark_1），构造连续 64 个“每步 1k 事件”的 debug 数据集，并证明事件、时间、TTC 标签和 mask 完全对齐。

请在 `EV-TTC-SNN-main` 下新建：

```text
snn_ttc/
├── data/
│   ├── build_1k_step_index.py
│   └── m3ed_geometry.py
└── tools/
    ├── make_1k_ttc_debug_set.py
    ├── audit_1k_ttc_debug_set.py
    └── visualize_1k_ttc_alignment.py
```

## 一、build_1k_step_index.py

直接读取 M3ED 原始左事件相机：

```text
prophesee/left/x
prophesee/left/y
prophesee/left/t
prophesee/left/p
```

按事件索引严格连续划分：

```text
step 0: [0,1000)
step 1: [1000,2000)
……
```

每一步记录：

```text
sequence_name
step_index
event_start_idx
event_end_idx
t_start
t_end
dt
is_sequence_start
```

必须断言：

```text
event_end_idx - event_start_idx == 1000
当前 step 的 event_start_idx 等于上一步 event_end_idx
t_start、t_end 单调递增
```

## 二、m3ed_geometry.py

从 EV-TTC 的 `calc_gt.py` 中拆出并封装：

1. 相机标定读取；
2. SE(3) 位姿插值；
3. 深度点云重投影；
4. `T`、`Omega` 计算；
5. TTC 和 inverse TTC 生成。

第一版只生成“分片起点标签”，与 EV-TTC 保持一致：

```text
深度重投影到 t_start；
速度由 t_start 到 t_end 的相对位姿计算；
TTC = Z / (Tz + 1e-5)；
inverse TTC = max(0, Tz / Z)。
```

不要修改原 EV-TTC 源码，采用调用、复制并注明来源或重新封装的方式。

## 三、make_1k_ttc_debug_set.py

先使用：

```text
spot_outdoor_day_skatepark_1
```

只生成连续 64 步。

输出 debug H5，至少包含：

```text
event_start_idx       [64]
event_end_idx         [64]
t_start               [64]
t_end                 [64]
dt                    [64]
event_cnt             [64,2,360,360]
depth_start           [64,360,360]
ttc_start             [64,360,360]
inverse_ttc_start     [64,360,360]
valid_ttc_mask        [64,360,360]
T                     [64,3]
Omega                 [64,3]
speed_valid           [64]
omega_valid           [64]
supervise_valid       [64]
```

其中：

```text
supervise_valid =
速度条件满足
AND 角速度条件满足
AND 标签存在足够有效像素
```

注意：

```text
速度和角速度筛选不能修改 valid_ttc_mask；
valid_ttc_mask 只描述像素标签是否可靠；
筛选失败的时间步仍必须保存在 H5 中。
```

第一版不要随机裁剪和翻转，先保留 `360×360`，减少排错变量。

## 四、audit_1k_ttc_debug_set.py

生成审计报告，至少检查：

1. 64 步是否严格连续；
2. 每步是否恰好 1000 个事件；
3. 每步 `dt` 的最小值、最大值、均值和分布；
4. `T`、`Omega`、`Tz` 的范围；
5. `speed_valid`、`omega_valid`、`supervise_valid` 数量；
6. 每步 `valid_ttc_mask` 有效像素率；
7. TTC、inverse TTC 的有限值、正负值和分布；
8. `mask=1` 位置是否全部为有限标签；
9. 是否存在空标签、全零图或异常极值；
10. 连续 10 步中筛选失败的步骤仍然保留。

## 五、visualize_1k_ttc_alignment.py

连续选择 10 步，每一步保存：

```text
正事件计数图
负事件计数图
depth
TTC
inverse TTC
valid_ttc_mask
t_start、t_end、dt、Tz、supervise_valid
```

同时生成一个 10 步横向时序总览图，便于人工检查事件运动与 depth/TTC 变化是否一致。

## 六、完成后输出

1. 代码；
2. debug H5 路径；
3. `audit_report.md`；
4. 可视化图片目录；
5. `implementation_notes.md`，说明复用了 EV-TTC 哪些函数、做了哪些改动；
6. 运行命令和实际运行结果。

当前阶段不要实现 Dataset、模型或训练循环。
