# Car多事件数三种空间方案与EV-TTC官方对比指令

## 一、任务目标

基于已有 `car_urban_night_rittenhouse` 数据，继续比较：

```text
N = 5k、10k、15k、20k
```

并针对每种N构造三种空间输入：

```text
S1：360×360全图
S2：360×360下采样至180×180，再中心裁剪128×128
S3：360×360上随机裁剪128×128
```

同时与EV-TTC官方 `360×360 signed IIR + 固定10 ms标签` 进行对比。

本阶段优先做“同一时间窗、同一标签时刻、只改变空间处理”的公平消融。尽量复用已有H5和原始事件索引，不重新完整扫描原始数据。

---

## 二、已有数据复用

优先复用：

```text
1. EV-TTC官方360×360 exp_filts / ttcef
2. 当前N=5k、10k、15k、20k生成的H5
3. 当前H5中的：
   raw_event_start_idx
   raw_event_end_idx
   t_start
   t_end
   T
   Omega
   ROI坐标
   block信息
```

对每个已有step，只根据：

```text
[raw_event_start_idx, raw_event_end_idx)
```

重新读取该step对应的原始事件，并生成S1、S2、S3。不要从头重新搜索分片边界。

---

## 三、三种空间方案

### S1：360×360全图

```text
原始事件
→ EV-TTC同口径去畸变
→ 中央720×720
→ 下采样到360×360
→ 保留完整360×360
```

输出：

```text
event_cnt：[2,360,360]
depth：[360,360]
signed inverse TTC：[360,360]
mask：[360,360]
```

要求：

- 正负事件分通道；
- 四邻域双线性累积；
- 不使用指数衰减；
- 每个step沿用现有N分片的时间范围。

---

### S2：360→180→中心128

```text
360×360事件、depth、inverse TTC、mask
→ 下采样到180×180
→ 中心裁剪128×128
```

中心裁剪：

```text
x=[26:154)
y=[26:154)
```

对应原360视野约：

```text
x=[52:308)
y=[52:308)
```

即覆盖中央约256×256区域。

下采样要求：

```text
事件图：2×2 sum pooling
depth：只对有效像素做2×2均值或中值
signed inverse TTC：只对有效像素做2×2均值
mask：2×2内存在有效像素则置1
```

不要把无效0或NaN直接参加普通平均。

S2作为“较大视野、较低分辨率、128输入尺寸”的主候选。

---

### S3：360上随机裁剪128

沿用当前MAVLab式方案：

```text
360×360
→ 随机/九宫格裁剪128×128
```

要求：

- 同一10步block内ROI固定；
- block结束或timeout后切换ROI；
- 保存 `roi_x0、roi_y0`；
- ROI切换处 `reset_required=1`；
- 事件、depth、inverse TTC和mask使用相同ROI。

---

## 四、公平比较原则

### 4.1 三种空间方案必须使用同一时间窗

对同一个step，S1、S2、S3必须共享：

```text
raw_event_start_idx
raw_event_end_idx
t_start
t_end
T
Omega
```

即：

```text
只改变空间处理
不改变事件时间窗
不改变标签运动窗口
```

主标签继续使用：

```text
运动窗口 = [t_start,t_end]
depth时刻 = t_start
signed inverse TTC = Tz / Z
```

保留正负signed inverse TTC。

主mask只要求：

```text
depth有效
重投影有效
inverse TTC有限
事件活跃
```

额外生成：

```text
positive_ttc_mask = main_mask AND inverse_TTC > 0
```

用于后续避障分析。

### 4.2 禁止跨空间方案重新累计N事件

本次不能让：

```text
S1在全图重新累计N
S2在中心区域重新累计N
S3在随机ROI重新累计N
```

否则时间尺度也会改变。

正确做法：

```text
先固定已有N分片的[t_start,t_end]
再对该时间窗内事件分别编码为S1、S2、S3
```

---

## 五、EV-TTC官方对比

### 5.1 官方原生数据

直接读取：

```text
360×360 signed IIR
固定10 ms运动窗口
官方TTC
官方mask
```

用于官方输入、标签分布和定性可视化。

### 5.2 同时刻固定10 ms公平标签

不能只匹配最近官方样本做数值比较。

对每个抽样step，在相同 `t_start` 上额外计算：

```text
运动窗口：[t_start,t_start+10ms]
depth时刻：t_start
```

生成：

```text
T_fixed10ms
Omega_fixed10ms
signed_inverse_ttc_fixed10ms
official_style_mask
positive_ttc_mask_fixed10ms
```

用于比较：

```text
事件窗口标签 vs 固定10 ms标签
```

从而消除最近邻官方样本带来的毫秒级时间误差。

### 5.3 输入表征对比

在相同或最近官方时间点可视化：

```text
官方signed IIR
S1固定事件计数
S2固定事件计数
S3固定事件计数
```

该部分只做定性比较，不对不同时间窗输入直接计算像素MAE。

---

## 六、生成规模

为避免再次长时间完整生成，分两阶段。

### 阶段A：配对探索集

每种N抽取：

```text
至少500个完整10步block
```

不足500则使用全部。

抽样覆盖：

```text
不同时间段
短/中/长event_dt
正inverse TTC
负inverse TTC
低风险
高风险
不同ROI
角速度通过/失败
```

每种N约5000个step，四种N合计约20000个step。

对这些step完整生成S1、S2、S3。

### 阶段B：正式数据生成

阶段A完成后，再根据结果选择一个或两个空间方案生成完整训练H5。当前任务不要默认三种方案全部完整生成。

---

## 七、输出H5设计

建议目录：

```text
EV-TTC-SNN-main/debug_sets/car_spatial_ablation/
├── N5000_spatial_compare.h5
├── N10000_spatial_compare.h5
├── N15000_spatial_compare.h5
└── N20000_spatial_compare.h5
```

每个step保存：

```text
N
step_index
block_index
step_in_block
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
T_event_window
Omega_event_window
T_fixed10ms
Omega_fixed10ms
roi_x0
roi_y0
reset_required
```

空间数据：

```text
event_cnt_360
event_cnt_180_center128
event_cnt_random128

inverse_ttc_360
inverse_ttc_180_center128
inverse_ttc_random128

mask_360
mask_180_center128
mask_random128

positive_mask_360
positive_mask_180_center128
positive_mask_random128
```

depth只需在审计子集中完整保存。

---

## 八、空间一致性检查

自动检查：

```text
S2中心裁剪与S1对应区域一致
S3 ROI与S1对应区域一致
事件、depth、inverse TTC、mask使用同一几何变换
不存在上下翻转、左右翻转或坐标偏移
```

S3检查：

```text
从S1直接裁剪的标签
vs
现有S3标签
```

统计：

```text
inverse TTC MAE
mask IoU
depth MAE
```

S2检查：

```text
S1按规定下采样并裁剪
vs
独立生成的S2
```

应仅有浮点误差。

---

## 九、输入与标签统计

对每个N、每种空间方案统计：

### 输入事件

```text
事件权重总和
非零像素率
正负事件比例
每像素事件值P50/P95/P99
最大像素事件值
空间稀疏度
```

### 标签

```text
mask有效像素率
signed inverse TTC分布
正值比例
负值比例
零附近比例
高风险像素比例
```

### 视野信息

统计或人工标注：

```text
是否包含完整道路结构
是否包含车辆或主要障碍
近场区域占比
高inverse TTC区域占比
```

---

## 十、计算量估计

对同一候选SNN估算：

```text
输入张量大小
第一层计算量
中间特征显存
单步前向时间
10步BPTT显存
10步BPTT耗时
```

注意：

```text
360×360面积约为128×128的7.91倍
```

S2与S3输入尺寸相同，重点比较：

```text
大视野低分辨率
vs
局部视野高分辨率
```

---

## 十一、综合对比表

每种N生成：

| 指标 | S1 360全图 | S2 180→中心128 | S3 随机128 | 官方EV-TTC |
|---|---:|---:|---:|---:|
| 输入尺寸 | 360×360 | 128×128 | 128×128 | 360×360 |
| 覆盖视野 | 全图 | 中央约256×256 | 局部128×128 | 全图 |
| event_dt P50 | 同N一致 | 同N一致 | 同N一致 | 10 ms |
| 事件非零率 |  |  |  |  |
| mask有效率 |  |  |  |  |
| inv-TTC正值比例 |  |  |  |  |
| inv-TTC负值比例 |  |  |  |  |
| 高风险像素比例 |  |  |  |  |
| 存储量 |  |  |  |  |
| BPTT显存估计 |  |  |  |  |
| 上下文完整性 |  |  |  |  |

再生成四种N总表，分析N与空间方案是否存在耦合。

---

## 十二、可视化要求

每种N、每种空间方案至少生成50组配对图。

每组显示：

```text
S1/S2/S3事件
S1/S2/S3 signed inverse TTC
S1/S2/S3 mask
官方IIR
官方TTC
官方mask
```

在S1和官方全图上画出：

```text
S2中央视野框
S3随机ROI黄色框
```

标注：

```text
N
t_start/t_end
event_dt
Tz
||T||
||Omega||
ROI坐标
正负inverse TTC比例
与官方时刻差
```

---

## 十三、脚本建议

新增：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── build_car_spatial_ablation_from_existing_h5.py
├── downsample_360_to_center128.py
├── audit_car_spatial_alignment.py
├── compare_car_spatial_schemes.py
├── estimate_snn_spatial_compute.py
└── visualize_car_spatial_ablation.py
```

核心参数：

```text
--source-dir <现有5k/10k/15k/20k目录>
--raw-data <car data.h5>
--event-counts 5000 10000 15000 20000
--blocks-per-n 500
--seed 42
--resume
```

---

## 十四、输出文件

至少输出：

1. `Car三种空间方案数据生成说明.md`
2. `Car三种空间方案空间对齐审计.md`
3. `Car三种空间方案事件统计.md`
4. `Car三种空间方案标签统计.md`
5. `Car三种空间方案计算量对比.md`
6. `Car多事件数与EVTTC官方综合对比.md`
7. 四个配对探索H5
8. step级统计CSV
9. 可视化目录
10. 日志目录
11. 运行时间和磁盘占用
12. 下一步训练推荐方案

---

## 十五、最终必须回答

1. 随机128是否明显丢失TTC所需上下文；
2. 180→中心128是否在同输入尺寸下保留更多有效视野；
3. 360全图能增加多少标签覆盖和风险信息；
4. 三种方案的事件稀疏度差异；
5. 空间对齐是否正确；
6. 5k、10k、15k、20k分别适合哪种空间方案；
7. S2与S3谁更适合作为第一版SNN输入；
8. 360全图是否值得承担约7.9倍输入面积；
9. 与EV-TTC官方差异主要来自表征、时间窗口还是空间视野；
10. 下一步优先训练哪种 `N + 空间方案`。
