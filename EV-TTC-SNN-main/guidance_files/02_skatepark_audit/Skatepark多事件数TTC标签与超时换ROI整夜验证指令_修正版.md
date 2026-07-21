# Skatepark多事件数TTC标签与超时换ROI整夜验证指令（修正版）

## 一、总体目标

在完整序列：

```text
spot_outdoor_day_skatepark_1
```

上独立验证：

```text
N = 5k、10k、15k、20k
```

每个SNN时间步由当前固定 `128×128` ROI 内连续出现的N条源事件构成。

同时加入：

```text
若当前ROI累计N条事件所需时间超过10 ms，则提前终止当前ROI并更换ROI
```

本次除统计事件时间尺度外，还要生成与事件分片严格对齐的TTC标签，并完成训练前的数据质量、标签口径、时间连续性和ROI敏感性验证。

要求采用断点续跑方式。单个N失败时，不终止其余N。

---

## 二、空间处理与事件编码

沿用已审计通过的EV-TTC空间处理：

```text
M3ED左事件相机原始事件
→ 去畸变
→ 中央720×720区域
→ 下采样到360×360
→ 选择固定128×128 ROI
```

要求：

- ROI尺寸始终为 `128×128`；
- 事件使用四邻域双线性分配；
- 正、负极性分别累积为两个非负通道；
- `event_cnt.shape = [2,128,128]`；
- `event_cnt.dtype = float32`；
- 一个源事件分配到多个像素后仍只计为一条源事件；
- 每个有效step必须恰好累计N条进入当前ROI的源事件。

---

## 三、ROI切换机制

### 3.1 正常切换

同一个ROI最多连续生成10个有效step：

```text
step 1～10：ROI_A
step 11～20：ROI_B
……
```

完成10个有效step后切换ROI，并记录：

```text
reset_required = 1
reset_reason = normal_roi_change
```

### 3.2 超时换ROI

统一设置：

```text
max_step_duration = 10 ms
```

若当前ROI从step起点开始，在10 ms内仍未累计到N条源事件：

1. 当前候选step记为 `timeout`；
2. 不将其作为有效训练step；
3. 记录当前已累计事件数和完成比例；
4. 丢弃该未完成step；
5. 立即终止当前ROI block；
6. 更换ROI；
7. 从当前原始事件位置之后继续扫描；
8. 不回退、不重叠、不重复使用原始事件；
9. 新ROI从零开始累计新step；
10. 记录：

```text
reset_required = 1
reset_reason = timeout_roi_change
```

超时前已完成的step仍保留在统计中；若该block不足10步，则标记为不完整block，不作为标准10步BPTT训练块。

### 3.3 ROI候选位置

使用固定3×3集合：

```text
x0 ∈ {0,116,232}
y0 ∈ {0,116,232}
```

共9个ROI。

要求：

- 第一块使用中心ROI `(116,116)`；
- 后续按固定随机种子打乱并循环；
- 四种N使用相同候选集合和随机种子；
- 每次切换不能与上一个ROI相同；
- 保存实际ROI切换轨迹。

---

## 四、TTC主标签口径

### 4.1 主标签必须使用事件分片起点与终点

对每个有效step，事件分片时间为：

```text
[t_start, t_end]
```

主标签的运动估计窗口也必须为：

```text
[t_start, t_end]
```

即：

```text
事件输入窗口 = 速度估计窗口
```

具体流程：

1. 在 `t_start` 时刻生成或重投影深度：

```text
depth_start = Z(t_start)
```

2. 根据 `t_start` 与 `t_end` 的相对位姿计算分片内平均运动：

```text
T_event_window
Omega_event_window
```

3. 使用事件相机坐标系下的前向速度 `Tz` 生成：

```text
TTC = Z / (Tz + eps)
inverse_TTC = Tz / Z
```

因此正式H5中的主标签为：

```text
T
Omega
ttc_start
inverse_ttc_start
valid_ttc_mask
```

其中 `T`、`Omega` 均来自当前事件分片 `[t_start,t_end]`。

### 4.2 固定10 ms标签只作为附加审计

从每种N中抽取至少1000个有效step，额外计算：

```text
运动窗口 = [t_start, t_start + 10 ms]
```

生成：

```text
T_fixed10ms
Omega_fixed10ms
TTC_fixed10ms
inverse_TTC_fixed10ms
```

这些字段只用于比较，不作为主训练标签。

重点比较：

```text
||T_event_window - T_fixed10ms||
||Omega_event_window - Omega_fixed10ms||
inverse TTC MAE
inverse TTC MRE
不同event_dt区间下的标签差异
```

目的：判断不同N对应的事件窗口是否足够长，是否会导致位姿插值和速度估计不稳定。

---

## 五、像素级标签有效性

主标签仅保留满足以下条件的像素：

```text
depth有限且为正
TTC有限
inverse TTC有限
TTC > 0
inverse TTC > 0
重投影有效
```

以下像素全部在 `valid_ttc_mask` 中置0：

```text
无效深度
重投影失败
Tz <= 0导致的负TTC
TTC <= 0
inverse TTC <= 0
NaN
Inf
```

不要将负TTC取绝对值。

---

## 六、时间步级筛选

继续记录：

```text
speed_valid
omega_valid
supervise_valid
```

阈值沿用当前方案：

```text
spot序列速度阈值：||T|| > 0.25 m/s
角速度阈值：||Omega|| < 0.18 rad/s
supervise_valid = speed_valid AND omega_valid
```

其中 `T` 和 `Omega` 必须使用当前事件窗口 `[t_start,t_end]` 计算。

筛选失败的step：

- 不从连续事件流中删除；
- 不计算该step的直接监督损失；
- 仍可参与SNN状态演化；
- 不因筛选失败自动reset；
- 只有ROI切换或timeout才reset。

---

## 七、四组独立实验

分别执行：

```text
N=5000
N=10000
N=15000
N=20000
```

每组均扫描完整序列，并保持以下参数一致：

```text
ROI尺寸
ROI候选集合
随机种子
每ROI最多10个有效step
10 ms超时阈值
主标签使用[t_start,t_end]
```

不同N允许形成不同的实际ROI切换轨迹。

---

## 八、完整BPTT block定义

标准10步BPTT block必须满足：

```text
同一ROI
连续10个有效输入step
中间无timeout
中间无ROI切换
```

注意：

- `supervise_valid=0` 的step可以存在于完整block中；
- 只要事件连续、ROI不变且没有timeout，就保持状态连续；
- 是否计算监督损失由 `supervise_valid` 决定；
- block开始处设置 `reset_required=1`；
- block内部设置 `reset_required=0`。

分别记录：

```text
all_input_steps
complete_10step_blocks
incomplete_blocks
supervised_steps
unsupervised_steps
```

---

## 九、H5输出设计

每种N单独输出一个H5：

```text
EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc/
├── skatepark_N5000.h5
├── skatepark_N10000.h5
├── skatepark_N15000.h5
└── skatepark_N20000.h5
```

### 9.1 每个有效step保存

```text
sequence_name
N
step_index
block_index
step_in_block
roi_x0
roi_y0
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
raw_event_index_span
roi_source_event_count
total_mapped_weight
event_cnt
T
Omega
speed_valid
omega_valid
supervise_valid
reset_required
reset_reason
inverse_ttc_start
valid_ttc_mask
```

其中：

```text
T、Omega、inverse_ttc_start
```

必须来自 `[t_start,t_end]`。

### 9.2 可选冗余字段

为降低磁盘占用，主H5可不保存完整：

```text
depth_start
ttc_start
```

优先保存：

```text
inverse_ttc_start float32
valid_ttc_mask uint8
```

仅在有效像素上由：

```text
ttc_start = 1 / inverse_ttc_start
```

恢复。

### 9.3 审计子集H5

每种N至少抽取100个完整10步block，保存：

```text
event_cnt
depth_start
ttc_start
inverse_ttc_start
valid_ttc_mask
T
Omega
T_fixed10ms
Omega_fixed10ms
TTC_fixed10ms
inverse_TTC_fixed10ms
```

抽样应覆盖：

```text
短event_dt
中位event_dt
接近10 ms的event_dt
低TTC
高TTC
不同ROI
supervise_valid=0
接近timeout
```

---

## 十、压缩、断点续跑与容错

使用：

```text
chunked HDF5
gzip或lzf压缩
逐block写盘
```

要求：

- 每完成一定数量block写checkpoint；
- 保存最后原始事件索引、ROI状态、block状态和随机数状态；
- 支持 `--resume`；
- 中断后不得从头开始；
- 每种N独立日志；
- 单个N失败后继续下一个N；
- 每个N完成后立即执行基础审计并落盘。

运行前估算磁盘空间。若空间不足，按以下优先级降级：

1. 必须保存全部标量索引和筛选字段；
2. 必须保存全部 `inverse_ttc_start + valid_ttc_mask`；
3. `event_cnt` 可仅保存完整训练block；
4. 完整depth和TTC仅保存审计子集；
5. 在报告中明确记录未保存字段。

---

## 十一、事件时间尺度统计

每种N统计：

```text
有效step数量
event_dt mean/std
min/max
P1/P5/P25/P50/P75/P95/P99
```

阈值比例：

```text
event_dt < 0.5 ms
event_dt < 1 ms
event_dt < 3.3 ms
event_dt < 7 ms
event_dt <= 10 ms
```

对完整10步block统计：

```text
block_duration = 第10步t_end - 第1步t_start
```

输出：

```text
mean/std
min/max
P5/P50/P95
```

以及：

```text
block_duration < 3.3 ms
block_duration < 7 ms
block_duration < 10 ms
block_duration < 30 ms
block_duration < 100 ms
```

---

## 十二、timeout与数据利用率统计

每种N统计：

```text
timeout数量
timeout比例
正常ROI切换次数
超时ROI切换次数
完整block比例
不完整block比例
超时时已累计事件数
completion_ratio
连续timeout最大次数
timeout丢弃事件总数
有效事件利用率
```

按9个ROI分别统计：

```text
有效step数
timeout比例
event_dt P50/P95
完整block比例
```

---

## 十三、TTC标签质量统计

### 13.1 标签有效性

每种N统计：

```text
valid_ttc_mask有效像素率
空标签step数量
NaN/Inf数量
TTC正值/负值/零值数量
inverse TTC正值/负值/零值数量
```

自动检查：

```python
assert mask=1位置的depth有限且>0
assert mask=1位置的TTC有限且>0
assert mask=1位置的inverse_TTC有限且>0
```

### 13.2 标签分布

在mask内统计：

```text
TTC min/max/mean/std/P1/P5/P25/P50/P75/P95/P99
inverse TTC min/max/mean/std/P1/P5/P25/P50/P75/P95/P99
depth分布（审计子集）
T、Tz、||T||、Omega、||Omega||分布
```

### 13.3 风险区间统计

统计每个step中以下像素比例：

```text
TTC < 0.5 s
TTC < 1 s
TTC < 2 s
TTC < 3 s
TTC < 5 s
inverse TTC > 0.2 1/s
inverse TTC > 0.5 1/s
inverse TTC > 1.0 1/s
```

统计包含近场高风险像素的step比例。

### 13.4 时间连续性

同一完整block内部统计：

```text
相邻step depth MAE
相邻step TTC MAE
相邻step inverse TTC MAE
相邻step valid mask IoU
T相邻变化
Omega相邻变化
```

ROI切换边界不计算连续性指标。

---

## 十四、主标签与固定10 ms标签对比

在审计子集中，按 `event_dt` 分组：

```text
0～1 ms
1～3.3 ms
3.3～7 ms
7～10 ms
```

分别统计：

```text
T差异
Omega差异
inverse TTC MAE
inverse TTC MRE
TTC MAE
有效mask一致率
```

重点回答：

1. event_dt很短时，事件窗口速度是否明显偏离固定10 ms速度；
2. N增大后，两种标签是否趋于一致；
3. 哪个N开始具备稳定的分片内速度估计；
4. 是否存在因位姿插值精度导致的异常step。

---

## 十五、事件与标签空间对齐验证

每种N至少可视化10个完整block，共100步。

抽样覆盖：

```text
中心ROI
边缘ROI
短event_dt
长event_dt
低TTC
高TTC
高正事件比例
高负事件比例
接近timeout
supervise_valid=0
```

每步保存：

```text
正事件通道
负事件通道
正负叠加图
depth
TTC
inverse TTC
valid_ttc_mask
```

标注：

```text
N
block/step
ROI坐标
t_start/t_end
event_dt
Tz
||T||
||Omega||
supervise_valid
valid mask比例
最小TTC
距离timeout剩余时间
```

生成10步时序总览，人工检查：

- 事件边缘与depth/TTC结构是否大体对齐；
- ROI裁剪是否一致；
- 是否存在翻转、错位或缩放错误；
- block内标签变化是否合理。

---

## 十六、四种N综合比较

生成总表：

| 指标 | 5k | 10k | 15k | 20k |
|---|---:|---:|---:|---:|
| 有效step数 |  |  |  |  |
| event_dt P50 |  |  |  |  |
| event_dt P95 |  |  |  |  |
| timeout比例 |  |  |  |  |
| 完整block比例 |  |  |  |  |
| block duration P50 |  |  |  |  |
| block duration P95 |  |  |  |  |
| supervise_valid比例 |  |  |  |  |
| mask有效像素率 |  |  |  |  |
| TTC<1s step比例 |  |  |  |  |
| inverse TTC相邻变化 |  |  |  |  |
| 固定10 ms标签差异 |  |  |  |  |
| 数据利用率 |  |  |  |  |
| H5大小 |  |  |  |  |
| 总耗时 |  |  |  |  |

重点回答：

1. 哪个N能使单步主要覆盖 `1～10 ms`；
2. 哪个N的10步BPTT时间跨度合理；
3. 哪个N的timeout率与不完整block率可接受；
4. 哪个N能提供明显但连续的TTC变化；
5. 哪个N的事件窗口速度估计开始稳定；
6. 哪个N最适合正式SNN训练；
7. 是否需要采用“事件数N + 10 ms时间上限”的联合窗口。

---

## 十七、脚本与运行方式

建议新增：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── build_skatepark_multi_n_ttc.py
├── audit_skatepark_multi_n_ttc.py
├── compare_event_window_vs_fixed10ms.py
├── visualize_skatepark_multi_n_ttc.py
└── summarize_skatepark_multi_n_ttc.py
```

命令行至少支持：

```text
--sequence spot_outdoor_day_skatepark_1
--event-counts 5000 10000 15000 20000
--roi-size 128
--steps-per-roi 10
--max-step-duration-ms 10
--audit-fixed-window-ms 10
--seed 42
--resume
```

执行顺序：

```text
1. 预检查数据、位姿范围、深度范围和磁盘空间
2. N=5k
3. N=10k
4. N=15k
5. N=20k
6. 标签审计
7. 事件窗口与固定10 ms标签比较
8. 可视化
9. 汇总报告
```

所有命令写入：

```text
run_all_skatepark_multi_n_ttc.sh
```

脚本要求：

```text
set -o pipefail
每个N独立日志
失败后记录错误并继续下一个N
支持断点续跑
完成后生成总状态报告
```

---

## 十八、最终输出

至少输出：

1. 全部新增和修改源码；
2. `run_all_skatepark_multi_n_ttc.sh`；
3. 四个N对应的H5或分块H5；
4. 四个N的原始标量统计CSV；
5. `Skatepark多事件数TTC标签综合报告.md`；
6. `事件窗口与固定10ms标签对比.md`；
7. `超时换ROI机制分析.md`；
8. `TTC标签质量审计.md`；
9. `事件与TTC空间对齐审计.md`；
10. 可视化目录；
11. 日志目录；
12. 磁盘占用、运行时间和失败项说明；
13. 对 `5k/10k/15k/20k` 的最终排序；
14. 正式训练首选N及备选方案。

当前阶段不要修改SNN网络结构，不要开始正式训练。先完成数据和标签验证。
