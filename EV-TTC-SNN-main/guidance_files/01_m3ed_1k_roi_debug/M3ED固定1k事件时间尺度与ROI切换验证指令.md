# M3ED固定1k事件分片时间尺度统计验证指令

## 一、验证目标

验证“固定ROI内每1000个事件作为一个SNN时间步”是否适合M3ED。

当前只分析事件分片对应的物理时间长度，不生成depth、TTC或网络输入标签，避免无关计算。

重点回答：

1. M3ED中固定ROI内1k事件通常对应多长时间；
2. 连续10个时间步实际覆盖多长物理时间；
3. 不同序列、不同平台和不同ROI位置之间差异多大；
4. 若1k事件时间跨度过短，应增加每步事件数、增加BPTT长度，还是改用固定时间窗。

---

## 二、基本空间处理

沿用当前已验证的事件映射流程：

```text
M3ED原始事件
→ 左事件相机去畸变
→ 中央720×720区域
→ 2倍下采样到360×360
→ 在360×360中选择128×128 ROI
```

ROI尺寸始终固定为：

```text
128×128
```

每个时间步由当前ROI内连续出现的1000条源事件构成。

双线性分配后，一个源事件即使贡献到多个像素，也仍只计为1条源事件。

---

## 三、ROI切换规则

每连续10个时间步组成一个分析块：

```text
时间步1～10：使用ROI_A
时间步11～20：更换为ROI_B
时间步21～30：更换为ROI_C
……
```

要求：

1. ROI尺寸始终保持128×128；
2. 同一个10步块内ROI位置固定；
3. 每完成10步后重新选择ROI位置；
4. ROI必须完全位于360×360范围内；
5. 记录每个10步块的ROI左上角坐标`roi_x0、roi_y0`；
6. ROI切换后重新从原始事件流继续向后扫描，不允许时间倒退、重叠或重复使用事件；
7. 新ROI的第一个时间步从ROI切换时刻之后出现的事件开始累计；
8. 每个10步块之间视为新的空间序列，后续用于SNN训练时应在ROI切换处执行`reset_states()`。

ROI选择建议：

- 第一块使用中心ROI；
- 后续ROI可在合法范围内随机采样；
- 固定随机种子，保证结果可复现；
- 同时避免所有ROI都集中于同一区域。

合法范围：

```text
0 <= roi_x0 <= 232
0 <= roi_y0 <= 232
```

因为：

```text
360 - 128 = 232
```

---

## 四、完整扫描spot_outdoor_day_skatepark_1

首先完整读取：

```text
spot_outdoor_day_skatepark_1
```

从可生成有效事件映射的位置开始，一直扫描到序列末尾。

对每个时间步记录：

```text
sequence_name
block_index
step_index
step_in_block
roi_x0
roi_y0
roi_source_event_count
raw_event_start_idx
raw_event_end_idx
t_start
t_end
dt
raw_event_index_span
total_mapped_weight
```

其中：

```text
roi_source_event_count = 1000
dt = t_end - t_start
```

对每个10步块额外记录：

```text
block_t_start
block_t_end
block_duration
block_roi_x0
block_roi_y0
```

并计算：

```text
block_duration = 第10步t_end - 第1步t_start
```

---

## 五、扩展到7train其余序列

从当前EV-TTC 7train配置或现有合并脚本中自动读取训练序列名单，不要手工重复维护列表。

对7个训练序列全部执行相同流程：

```text
固定128×128 ROI
每10步更换一次ROI
每步固定ROI内1000条源事件
完整扫描整个序列
```

输出：

1. 每个序列单独的统计结果；
2. 7个序列汇总结果；
3. Spot序列汇总结果；
4. Car序列汇总结果。

---

## 六、每个序列的统计指标

### 1. 单步1k事件时间长度

统计：

```text
分片总数
dt最小值
dt最大值
dt均值
dt标准差
P1
P5
P25
P50
P75
P95
P99
```

单位统一输出为：

```text
微秒
毫秒
```

同时计算事件率：

```text
event_rate = 1000 / dt
```

### 2. 不同时间阈值的占比

统计单步`dt`满足以下条件的比例：

```text
dt < 0.1 ms
dt < 0.5 ms
dt < 1.0 ms
dt < 3.3 ms
dt < 7.0 ms
dt < 10.0 ms
```

### 3. 10步BPTT物理时间跨度

对每个10步块统计：

```text
block_duration
```

输出：

```text
均值
标准差
P5
P50
P95
最小值
最大值
```

并统计：

```text
block_duration < 1 ms
block_duration < 3.3 ms
block_duration < 7 ms
block_duration < 10 ms
```

的比例。

### 4. 原始事件索引跨度

由于只统计ROI内事件，每1000个ROI事件在原始全图事件流中可能跨越很多条记录。

统计：

```text
raw_event_index_span
```

的均值、标准差和分位数。

### 5. ROI位置敏感性

对不同ROI位置分别统计：

```text
每步dt均值
每步dt中位数
10步block_duration
事件率
```

生成ROI位置与时间跨度之间的关系图，检查中心区域是否明显高事件率。

---

## 七、反向估计目标时间窗所需事件数

针对目标时间窗：

```text
1 ms
3.3 ms
7 ms
10 ms
```

统计每个序列、每种平台和所有数据汇总情况下，ROI内平均包含多少源事件：

```text
N_1ms
N_3.3ms
N_7ms
N_10ms
```

同时输出中位数、P5和P95。

该结果用于回答：

```text
若希望一个SNN时间步覆盖约3.3 ms、7 ms或10 ms，
M3ED中每步应使用多少事件。
```

---

## 八、图表要求

每个序列生成：

1. 单步`dt`直方图；
2. 单步`dt`累积分布CDF；
3. 10步`block_duration`直方图；
4. 10步`block_duration`CDF；
5. `dt`随时间步变化曲线；
6. ROI位置与`dt`均值的散点图或热力图；
7. 每10步ROI切换示意图。

汇总生成：

1. 7个序列`dt`箱线图；
2. 7个序列10步时长箱线图；
3. Spot与Car对比图；
4. 不同目标时间窗对应事件数统计图。

---

## 九、判断原则

不要预设固定1k事件一定不可取，应根据数据得出结论。

若结果显示：

```text
绝大多数单步dt明显小于1 ms；
10步BPTT仍只覆盖数毫秒；
不同ROI和序列下均存在相同趋势；
```

则可以认为：

```text
MAVLab的N=1k、K=10不能原样迁移到M3ED。
```

后续应比较：

```text
方案A：增大每步事件数N
方案B：增大BPTT长度K
方案C：改用固定时间窗
```

若不同ROI或序列差异很大，则需要考虑：

```text
自适应事件数
固定时间窗
事件数上限与下限联合约束
```

---

## 十、实现要求

新建独立统计脚本，不修改现有debug H5生成代码。

建议文件：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── scan_roi_1k_dt_distribution.py
├── summarize_roi_1k_dt_distribution.py
└── plot_roi_1k_dt_distribution.py
```

扫描阶段只读取：

```text
事件x、y、t、p
相机标定参数
```

不读取或生成：

```text
depth
TTC
inverse TTC
pose
T
Omega
```

避免扫描全数据集时产生不必要开销。

---

## 十一、输出文件

完成后输出：

1. 完整源码；
2. 每个序列的原始统计CSV或H5；
3. `spot_outdoor_day_skatepark_1完整分片时间统计.md`；
4. `M3ED_7train固定1k事件时间尺度汇总.md`；
5. `ROI位置敏感性分析.md`；
6. `目标时间窗对应事件数量统计.md`；
7. 所有图表目录；
8. 实际运行命令；
9. 运行耗时和硬件占用；
10. 对固定1k事件是否适合M3ED的最终判断。

当前阶段不要生成TTC标签，不要修改网络，不要开始训练。
