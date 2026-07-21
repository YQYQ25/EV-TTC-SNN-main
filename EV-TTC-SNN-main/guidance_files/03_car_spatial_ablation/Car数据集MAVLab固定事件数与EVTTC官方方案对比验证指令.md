# Car数据集MAVLab式固定事件数与EV-TTC官方方案对比验证指令

## 一、验证目标

选择当前已下载的car训练序列：

```text
car_urban_night_rittenhouse
```

对同一序列分别采用：

```text
方案A：MAVLab式固定事件数输入
方案B：EV-TTC官方原生方法
方案C：同一ROI与同一时刻下的公平对齐方法
```

进行数据生成和统计比较。

重点回答：

1. car数据是否比Skatepark具有更高的角速度通过率；
2. car数据中 `Tz>0` 和正TTC样本比例是否更高；
3. 固定事件数 `5k、10k、15k、20k` 中哪个最适合SNN训练；
4. MAVLab式事件输入与EV-TTC官方固定10 ms方案之间的差异来自哪里；
5. 当前Skatepark监督稀疏是否主要是数据轨迹特性，而不是实现错误。

本阶段只完成数据生成、审计和比较，不开始正式训练。

---

## 二、数据与阈值

使用：

```text
car_urban_night_rittenhouse
```

确认以下文件存在：

```text
*_data.h5
*_depth_gt.h5
*_pose_gt.h5
```

car样本级筛选条件使用EV-TTC官方阈值：

```text
||T|| > 1.3 m/s
||Omega|| < 0.18 rad/s
```

同时额外定义：

```text
positive_ttc_valid = Tz > 0 AND valid_ttc_mask非空
direct_supervision_valid =
speed_valid
AND omega_valid
AND positive_ttc_valid
```

---

# 三、方案A：MAVLab式固定事件数

## 3.1 事件输入

分别生成：

```text
N = 5k、10k、15k、20k
```

空间流程：

```text
M3ED原始事件
→ 左事件相机去畸变
→ 中央720×720区域
→ 下采样到360×360
→ 固定128×128 ROI
→ 当前ROI内累计N条源事件
→ 正负极性双通道事件计数图
```

要求：

```text
event_cnt.shape = [2,128,128]
event_cnt.dtype = float32
```

正负极性分别累计，不相减。

## 3.2 ROI切换

沿用当前规则：

```text
同一ROI连续生成10个有效step
完成10步后更换ROI
```

ROI候选位置：

```text
x0 ∈ {0,116,232}
y0 ∈ {0,116,232}
```

共9个位置。

要求：

- 第一块使用中心ROI `(116,116)`；
- 后续按固定随机种子循环；
- ROI切换处记录 `reset_required=1`；
- block内部 `reset_required=0`。

## 3.3 超时规则

统一设置：

```text
max_step_duration = 10 ms
```

若10 ms内未累计到N条事件：

```text
当前step标记timeout
丢弃未完成step
结束当前block
更换ROI
继续向后扫描
不回退、不重复使用事件
```

记录：

```text
accumulated_event_count
completion_ratio
timeout_reason
reset_required
```

## 3.4 TTC主标签

每个step的主标签必须使用当前事件分片起止时间：

```text
事件窗口：[t_start,t_end]
运动窗口：[t_start,t_end]
深度时刻：t_start
```

计算：

```text
T
Omega
depth_start
TTC = Z / (Tz + eps)
inverse_TTC = Tz / Z
```

主训练mask只保留：

```text
depth有效
重投影有效
TTC > 0
inverse_TTC > 0
Tz > 0
finite
```

不要将负TTC取绝对值。

## 3.5 每种N保存

```text
event_cnt
inverse_ttc_start
valid_ttc_mask
T
Omega
speed_valid
omega_valid
supervise_valid
direct_supervision_valid
t_start
t_end
event_dt
ROI坐标
block信息
reset_required
timeout信息
```

---

# 四、方案B：EV-TTC官方原生方法

直接使用官方流程：

```text
create_exp.py
calc_gt.py
merge.py
```

要求保持官方口径：

```text
360×360全图
signed IIR事件表示
固定10 ms运动窗口
深度取运动起始时刻附近
car速度阈值1.3 m/s
角速度阈值0.18 rad/s
官方mask保留负TTC
```

统计：

```text
候选样本数
boundary有效样本数
速度通过率
角速度通过率
最终官方保留率
Tz>0比例
Tz<=0比例
正TTC样本比例
负TTC样本比例
official mask空样本比例
positive TTC mask空样本比例
```

另外离线构造：

```text
positive_ttc_mask =
official_mask
AND TTC > 0
AND inverse_TTC > 0
```

比较官方mask与只保留正TTC后的结果。

---

# 五、方案C：公平对齐方案

为了隔离不同因素，在方案A每个step的同一 `t_start` 上，额外生成以下标签：

## 5.1 固定10 ms标签

```text
运动窗口：[t_start,t_start+10ms]
空间范围：与方案A相同的128×128 ROI
深度时刻：t_start
```

生成：

```text
T_fixed10ms
Omega_fixed10ms
TTC_fixed10ms
inverse_TTC_fixed10ms
positive_ttc_mask_fixed10ms
```

与事件窗口标签比较：

```text
T差异
Omega差异
inverse TTC MAE
inverse TTC MRE
TTC MAE
mask一致率
```

## 5.2 同ROI的官方mask逻辑

在同一128×128 ROI上额外构造：

```text
official_style_mask
positive_ttc_mask
```

用于隔离：

```text
官方保留负TTC
当前只保留正TTC
```

## 5.3 同一时间点的全图与ROI比较

从抽样step中同时计算：

```text
360×360全图标签
128×128 ROI标签
```

比较：

```text
mask有效像素率
正TTC样本比例
空标签样本比例
TTC分布
```

---

# 六、完整10步Block监督密度

对方案A每种N统计：

```text
完整10步block数
不完整block数
timeout比例
每block direct supervision step数
```

输出0到10的完整直方图，并汇总：

```text
0监督block比例
1～3监督block比例
4～6监督block比例
7～9监督block比例
10监督block比例
每block平均监督step数
```

---

# 七、事件时间尺度统计

每种N统计：

```text
event_dt mean/std
P1/P5/P25/P50/P75/P95/P99
min/max
```

阈值比例：

```text
event_dt < 0.5 ms
event_dt < 1 ms
event_dt < 3.3 ms
event_dt < 7 ms
event_dt <= 10 ms
```

完整10步block统计：

```text
block_duration
P5/P50/P95
min/max
```

---

# 八、运动与筛选统计

对方案A和方案B分别统计：

```text
||T||
Tz
||Omega||
speed_valid
omega_valid
```

分类：

```text
A：速度通过、角速度通过
B：仅速度失败
C：仅角速度失败
D：速度和角速度同时失败
```

重点比较car与之前Skatepark结果：

```text
角速度通过率
Tz>0比例
正TTC有效率
direct supervision比例
```

---

# 九、TTC标签统计

分别统计：

```text
TTC > 0
TTC < 0
TTC = 0
inverse_TTC > 0
inverse_TTC < 0
mask空样本
mask有效像素率
```

风险区间：

```text
TTC < 0.5 s
TTC < 1 s
TTC < 2 s
TTC < 3 s
TTC < 5 s
inverse TTC > 0.2
inverse TTC > 0.5
inverse TTC > 1.0
```

---

# 十、综合比较表

生成：

| 指标 | 5k | 10k | 15k | 20k | 官方EV-TTC |
|---|---:|---:|---:|---:|---:|
| 候选step数 |  |  |  |  |  |
| event_dt P50 |  |  |  |  | 10 ms |
| event_dt P95 |  |  |  |  | 10 ms |
| 速度通过率 |  |  |  |  |  |
| 角速度通过率 |  |  |  |  |  |
| Tz>0比例 |  |  |  |  |  |
| 正TTC有效step比例 |  |  |  |  |  |
| direct supervision比例 |  |  |  |  |  |
| timeout比例 |  |  |  |  | — |
| 完整block比例 |  |  |  |  | — |
| 0监督block比例 |  |  |  |  | — |
| 10监督block比例 |  |  |  |  | — |
| mask有效像素率 |  |  |  |  |  |
| TTC<1s step比例 |  |  |  |  |  |
| 负TTC样本比例 |  |  |  |  |  |
| H5大小 |  |  |  |  |  |
| 运行时间 |  |  |  |  |  |

---

# 十一、与Skatepark对比

读取之前Skatepark统计，生成：

| 指标 | Skatepark 15k | Car 15k | Skatepark官方EV-TTC | Car官方EV-TTC |
|---|---:|---:|---:|---:|
| 速度通过率 |  |  |  |  |
| 角速度通过率 |  |  |  |  |
| Tz>0比例 |  |  |  |  |
| 正TTC有效率 |  |  |  |  |
| direct supervision比例 |  |  |  |  |
| 0监督block比例 |  |  | — | — |
| TTC<1s比例 |  |  |  |  |

重点判断：

```text
car是否明显优于Skatepark
```

---

# 十二、可视化

每种N至少抽取：

```text
10个完整block
```

覆盖：

```text
短event_dt
中位event_dt
接近10 ms
低TTC
高TTC
Tz>0
Tz<=0
角速度通过
角速度失败
```

每步保存：

```text
正事件通道
负事件通道
事件叠加图
depth
TTC
inverse TTC
positive mask
official-style mask
```

官方EV-TTC方案抽取相同类别样本进行对照。

---

# 十三、实现建议

新增脚本：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── build_car_multi_n_ttc.py
├── audit_car_multi_n_ttc.py
├── audit_official_evttc_car.py
├── compare_car_event_window_vs_fixed10ms.py
├── compare_car_roi_vs_official.py
└── visualize_car_comparison.py
```

命令行支持：

```text
--sequence car_urban_night_rittenhouse
--event-counts 5000 10000 15000 20000
--roi-size 128
--steps-per-roi 10
--max-step-duration-ms 10
--seed 42
--resume
```

---

# 十四、执行顺序

```text
1. 检查数据、磁盘和时间范围
2. 运行官方EV-TTC car审计
3. 生成5k
4. 生成10k
5. 生成15k
6. 生成20k
7. 生成固定10 ms公平对齐标签
8. 审计监督密度
9. 可视化
10. 生成综合报告
```

要求：

```text
支持断点续跑
每种N独立日志
单个任务失败后继续
完成后生成总状态报告
```

---

# 十五、输出文件

至少输出：

1. `Car多事件数TTC数据生成报告.md`
2. `Car官方EVTTC筛选与负TTC统计.md`
3. `Car事件窗口与固定10ms标签对比.md`
4. `Car多事件数监督密度统计.md`
5. `Car固定事件数与官方EVTTC综合对比.md`
6. `Car与Skatepark对比报告.md`
7. 四种N对应H5
8. 官方候选样本CSV
9. step级和block级CSV
10. 可视化目录
11. 日志目录
12. 运行时间和磁盘占用
13. 对5k、10k、15k、20k的最终排序
14. 正式训练推荐配置

---

# 十六、最终必须回答

1. car角速度通过率是否明显高于Skatepark；
2. car的 `Tz>0` 比例是否更高；
3. car正TTC有效样本比例是否更高；
4. car的0监督block比例是否明显下降；
5. 固定事件数方案中哪个N最合适；
6. 事件窗口标签与固定10 ms标签差异是否更小；
7. 官方EV-TTC与当前ROI方案的主要差异是什么；
8. 当前SNN-TTC正式训练应优先使用car、Spot还是混合数据；
9. 是否仍需要保留负TTC；
10. 下一步是否可以进入小规模训练。
