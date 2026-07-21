# SuperviseValid、空标签Step与10步Block监督密度审计指令

## 1. 目标

基于已生成的 `N=5k、10k、15k、20k` 四组 Skatepark TTC H5，查清：

1. `supervise_valid=0` 的具体原因；
2. 空标签 step 的具体原因；
3. 两者是否重合；
4. 每个完整10步 block 中实际有多少个可监督 step；
5. 哪个 N 的监督质量和监督密度更适合训练。

本阶段只审计现有数据，不重生成事件和TTC，不修改H5，不开始训练。

## 2. 审计对象

自动定位：

```text
EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc/
├── skatepark_N5000.h5
├── skatepark_N10000.h5
├── skatepark_N15000.h5
└── skatepark_N20000.h5
```

若文件名不同，记录实际路径。

## 3. supervise_valid=0 原因分类

当前定义：

```text
speed_valid = ||T|| > 0.25 m/s
omega_valid = ||Omega|| < 0.18 rad/s
supervise_valid = speed_valid AND omega_valid
```

逐step划分：

```text
A：speed_valid=1，omega_valid=1，可监督
B：speed_valid=0，omega_valid=1，仅速度失败
C：speed_valid=1，omega_valid=0，仅角速度失败
D：speed_valid=0，omega_valid=0，两者同时失败
```

每种N统计：

- A/B/C/D数量和比例；
- B/C/D占 `supervise_valid=0` 的比例；
- `||T||`、`Tz`、`||Omega||` 的 mean、P5、P25、P50、P75、P95；
- 阈值裕量：

```text
speed_margin = ||T|| - 0.25
omega_margin = 0.18 - ||Omega||
```

重点判断 `supervise_valid=0` 主要由速度还是角速度造成，是否存在大量接近阈值的step。

## 4. 空标签Step定义

```text
empty_label_step = valid_ttc_mask.sum() == 0
```

空标签与 `supervise_valid=0` 必须分开统计。

对每个step记录：

```text
valid_pixel_count
valid_pixel_ratio
empty_label_step
```

## 5. 空标签原因分类

对空标签step检查以下原因。

### 5.1 位姿或时间异常

```text
t_start/t_end超出pose范围
位姿插值失败
event_dt异常
相对位姿非法
```

标记：

```text
pose_or_time_invalid
```

### 5.2 深度为空

起始时刻ROI内：

```text
finite(depth_start) AND depth_start > 0
```

的像素数为0。

标记：

```text
depth_empty
```

### 5.3 重投影或ROI裁剪后为空

原始深度有效，但重投影到事件相机或裁剪到当前ROI后无有效像素。

分别标记：

```text
reprojection_empty
roi_crop_empty
```

### 5.4 前向速度无效

```text
Tz <= 0
```

标记：

```text
tz_nonpositive
```

### 5.5 数值异常

深度有效且 `Tz>0`，但TTC或inverse TTC全部为：

```text
NaN、Inf、TTC<=0、inverse_TTC<=0
```

标记：

```text
numeric_invalid
```

### 5.6 未知原因

无法归类时标记：

```text
unknown
```

一个step可同时具有多个原因标志，同时按以下优先级给出一个主原因：

```text
pose/time异常
→ depth为空
→ 重投影或ROI为空
→ Tz<=0
→ 数值异常
→ unknown
```

若主H5缺少 `depth_start` 等字段，使用审计子集或按保存的step索引从原始数据补算。不能确定时写 `unknown`，不要猜测。

## 6. supervise_valid与空标签交叉分析

每种N生成2×2表：

|  | 非空标签 | 空标签 |
|---|---:|---:|
| supervise_valid=1 |  |  |
| supervise_valid=0 |  |  |

计算：

```text
P(empty | supervise_valid=1)
P(empty | supervise_valid=0)
P(supervise_valid=0 | empty)
P(supervise_valid=0 | nonempty)
```

定义真正可直接监督的step：

```text
direct_supervision_valid =
supervise_valid AND (valid_ttc_mask.sum() > 0)
```

统计其数量和比例。

重点回答：

- 是否存在 `supervise_valid=1` 但空标签；
- 是否存在标签非空但 `supervise_valid=0`；
- 空标签是否主要由运动筛选失败造成。

## 7. 完整10步Block监督密度

只统计完整10步block。每个block计算：

```text
num_supervise_valid
num_nonempty_label
num_direct_supervision_valid
```

其中：

```text
num_direct_supervision_valid =
sum(supervise_valid AND nonempty_label)
```

输出0到10的完整直方图，并汇总：

```text
0个
1～3个
4～6个
7～9个
10个
```

每种N统计：

- 完整block总数；
- 每block平均、中位可直接监督step数；
- P5/P25/P75/P95；
- 0监督block比例；
- 1～3监督step block比例；
- 至少5个监督step block比例；
- 10步全部可监督block比例。

## 8. Block内位置和连续性

对 `step_in_block=0～9` 分别统计：

```text
supervise_valid比例
非空标签比例
direct_supervision_valid比例
```

并统计每个block内：

```text
最长连续可监督step数
最长连续无监督step数
可监督片段数量
```

用于判断监督信号是否过于零散。

## 9. 四种N横向比较

生成总表：

| 指标 | 5k | 10k | 15k | 20k |
|---|---:|---:|---:|---:|
| supervise_valid比例 |  |  |  |  |
| 仅速度失败比例 |  |  |  |  |
| 仅角速度失败比例 |  |  |  |  |
| 两者同时失败比例 |  |  |  |  |
| 空标签step比例 |  |  |  |  |
| Tz<=0空标签比例 |  |  |  |  |
| depth/reprojection空标签比例 |  |  |  |  |
| direct supervision比例 |  |  |  |  |
| 每block平均监督step数 |  |  |  |  |
| 0监督block比例 |  |  |  |  |
| ≥5监督step block比例 |  |  |  |  |
| 10监督step block比例 |  |  |  |  |

重点解释N增大后 `supervise_valid` 比例为何下降。

## 10. 异常样本导出

每种N导出以下样本的step索引：

```text
supervise_valid=1且空标签
supervise_valid=0且非空标签
完整block中0个监督step
完整block中10个监督step
Tz<=0空标签
depth/reprojection空标签
unknown空标签
```

每类保存前20个和随机20个，字段至少包括：

```text
N
step_index
block_index
step_in_block
ROI坐标
t_start/t_end/event_dt
||T||/Tz/||Omega||
speed_valid/omega_valid/supervise_valid
valid_pixel_ratio
empty_label_reason
```

已有可视化脚本时，每类随机可视化5个样本。

## 11. 实现建议

新建独立脚本：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── audit_supervise_and_empty_labels.py
├── audit_block_supervision_density.py
└── summarize_supervision_audit.py
```

支持：

```text
--input-dir
--event-counts 5000 10000 15000 20000
--output-dir
```

所有step级和block级结果同时保存为CSV。

## 12. 输出文件

至少生成：

```text
SuperviseValid失效原因统计.md
空标签Step原因统计.md
SuperviseValid与空标签交叉分析.md
完整10步Block监督密度统计.md
四种N监督质量综合对比.md
step级审计.csv
block级审计.csv
异常样本索引.csv
```

并附完整运行命令。

## 13. 报告必须明确回答

1. `supervise_valid=0` 主要由速度失败还是角速度失败造成；
2. 空标签step主要由什么原因造成；
3. 两者是否高度重合；
4. 真正可直接监督的step比例是多少；
5. 每个完整10步block平均有多少个监督step；
6. 0监督block比例是多少；
7. 哪个N的监督质量最好；
8. 是否需要调整速度或角速度阈值；
9. 是否需要修改采样策略；
10. 当前数据是否适合进入正式训练。
