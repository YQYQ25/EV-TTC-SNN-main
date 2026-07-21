# Car三种空间方案既有实验最小修正执行指令

## 一、任务目标

基于已经完成的：

```text
car_urban_night_rittenhouse
```

三种空间方案实验，进行最小修改和结果修正。

不要新开一套平行实验，不要覆盖旧数据，也不要立即对所有N重新生成完整H5。

本次需要完成：

1. 将旧实验正式标记为“同时间窗空间消融”；
2. 修正旧 `N=10000` 配对数据中的S2标签生成逻辑；
3. 一次扫描原始事件流，生成S1和S2在 `5k、10k、15k、20k` 下各自独立的固定N分片索引；
4. 先仅物化 `N=10000` 的S1和S2完整事件与标签；
5. 复用现有S3 `N=10000` H5，完成三种正确固定N方案的首轮对比；
6. 输出结果后停止，不自动扩展其余N完整H5。

---

# 二、三种方案的正确定义

## S1：360×360全图固定N

```text
在完整360×360有效视野内
连续累计N条源事件
```

每个step独立得到：

```text
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
T
Omega
depth
signed inverse TTC
mask
```

## S2：中央大视野固定N

S2输入最终为：

```text
128×128
```

但对应原360图中的中央：

```text
x=[52:308)
y=[52:308)
```

即约256×256视野。

正确事件定义：

```text
先判断事件是否落入360坐标中的中央256×256区域
在该区域内连续累计N条源事件
再将事件坐标按2倍下采样映射到128×128
```

因此S2也必须独立拥有自己的：

```text
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
T
Omega
depth
signed inverse TTC
mask
```

## S3：随机/九宫格128×128 ROI固定N

直接复用现有源H5：

```text
car_urban_night_rittenhouse_N5000.h5
car_urban_night_rittenhouse_N10000.h5
car_urban_night_rittenhouse_N15000.h5
car_urban_night_rittenhouse_N20000.h5
```

这些H5已经是真正的ROI内固定N数据，不重新生成。

---

# 三、冻结和标记旧实验

旧目录：

```text
EV-TTC-SNN-main/debug_sets/[5]car_spatial_ablation
```

不得删除或覆盖。

将旧实验明确标记为：

```text
paired_window_spatial_ablation
```

旧数据实际含义：

```text
S3随机128 ROI先累计N事件得到[t_start,t_end]
S1和S2沿用S3时间窗
```

因此旧S1和S2不是独立固定N方案。

给旧H5补充attributes：

```text
experiment_mode = paired_window_spatial_ablation
window_source = S3_random128
event_count_definition_S1 = variable_events_in_S3_window
event_count_definition_S2 = variable_events_in_S3_window
event_count_definition_S3 = fixed_N_in_random128
label_version = old_or_v2
```

旧报告中以下表述应标记失效：

```text
S1是360全图固定N
S2是中央大视野固定N
S1/S2/S3事件数一致
旧结果足以决定正式训练排序
```

---

# 四、先修正旧N=10000配对实验中的S2标签

这一步不重新扫描原始74.88 GiB事件流。

目的：

```text
验证S2标签破碎是否主要来自错误的标签池化与mask池化
```

## 4.1 旧S2事件输入

旧配对实验中的：

```text
event_cnt_180_center128
```

可继续保留，因为它只是同一S3时间窗下对S1事件做2×2 sum pooling并中心裁剪，适合作为paired-window辅助输入。

## 4.2 必须重算的字段

对 `N10000_spatial_compare.h5` 重算：

```text
depth_180_center128
inverse_ttc_180_center128
mask_180_center128
positive_mask_180_center128

inverse_ttc_fixed10ms_180_center128
mask_fixed10ms_180_center128
positive_mask_fixed10ms_180_center128
```

## 4.3 正确S2标签生成

不能再使用：

```text
S1 depth valid mean pooling
S1 inverse TTC valid mean pooling
S1最终mask any pooling
```

正确流程：

```text
t_start时刻原始depth
→ 直接投影到S2目标相机几何
→ 得到128×128 depth_S2
→ 使用当前分片Tz计算 signed inverse TTC
→ 根据S2事件活跃、深度有效、数值有限重新生成mask
```

S2等效相机内参：

```text
fx_S2 = fx_360 / 2
fy_S2 = fy_360 / 2
cx_S2 = cx_360 / 2 - 26
cy_S2 = cy_360 / 2 - 26
```

使用z-buffer或与EV-TTC一致的深度投影方式。

标签：

```text
inverse_TTC_S2 = Tz / depth_S2
```

mask：

```text
mask_S2 =
depth_valid
AND reprojection_valid
AND event_active_S2
AND finite(inverse_TTC_S2)
```

训练signed inverse TTC时不要增加：

```text
Tz > 0
inverse TTC > 0
```

额外生成：

```text
positive_mask_S2 = mask_S2 AND inverse_TTC_S2 > 0
```

## 4.4 修正后对比

生成旧版与新版S2对比：

```text
旧S2 inverse TTC
新版S2 inverse TTC
旧mask
新版mask
差值图
```

统计：

```text
inverse TTC MAE
mask IoU
mask有效率
局部连续性
孤立像素数量
高风险像素比例
```

至少重新输出之前发现破碎问题的典型样本。

---

# 五、一次raw pass生成S1/S2全部N独立分片索引

修改现有：

```text
build_car_spatial_ablation_from_existing_h5.py
```

优先复用：

```text
build_skatepark_multi_n_ttc.py
```

中的事件映射、位姿插值和H5写入逻辑。

不要按每个N和每个方案分别重复扫描原始事件文件。

一次顺序扫描中同时维护：

```text
S1：5k、10k、15k、20k
S2：5k、10k、15k、20k
```

共8个事件计数器。

## S1事件判定

事件完成去畸变并映射到360×360后，只要至少有有效贡献进入360范围，即计为S1源事件。

## S2事件判定

事件映射到360坐标后，若至少有有效贡献进入：

```text
x=[52:308)
y=[52:308)
```

则计为S2源事件。

一个源事件即使双线性贡献到多个像素，也只计为一条源事件。

## 分片输出

每个方案、每个N独立保存：

```text
scheme
N
step_index
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
source_event_count
raw_event_index_span
```

先只生成索引CSV或轻量H5，不物化事件图和TTC大数组。

要求：

```python
assert source_event_count == N
assert t_end > t_start
assert raw_event_end_idx > raw_event_start_idx
```

---

# 六、独立fixed-N主实验的block规则

S1和S2分别独立形成自己的连续step序列。

统一使用：

```text
每10个step组成一个BPTT block
```

S1和S2不需要ROI切换。

因此：

```text
每10步后不reset
```

是否reset应由真实序列边界决定，而不是人为每10步重置。

对于截断BPTT：

```text
每10步backward并detach_states
但不reset_states
```

只有以下情况reset：

```text
序列开始
真实时间断裂
数据边界
主动丢弃导致连续性破坏
```

注意：S3仍按ROI切换处reset。

## 超时规则

仍使用：

```text
max_step_duration = 10 ms
```

若当前区域10 ms内未累计到N条事件：

```text
记录timeout
丢弃未完成step
从当前时间继续累计下一step
```

S1/S2没有ROI可切换，因此不能套用“超时换ROI”。

请在报告中明确S1/S2 timeout后的处理逻辑，避免错误重置或时间回退。

---

# 七、先只物化N=10000的S1和S2

索引扫描完成后，先只生成：

```text
S1_N10000.h5
S2_N10000.h5
```

S3直接复用：

```text
car_urban_night_rittenhouse_N10000.h5
```

## 每个step保存

```text
scheme
N
step_index
block_index
step_in_block
raw_event_start_idx
raw_event_end_idx
t_start
t_end
event_dt
event_cnt
depth_start
signed_inverse_ttc
valid_ttc_mask
positive_ttc_mask
T
Omega
speed_valid
omega_valid
supervise_valid
timeout
reset_required
```

car筛选条件：

```text
||T|| > 1.3 m/s
||Omega|| < 0.18 rad/s
```

主标签保留正负signed inverse TTC。

---

# 八、N=10000三方案对比

正确对比对象：

```text
S1：全图内独立累计10k
S2：中央256视野内独立累计10k
S3：随机128 ROI内独立累计10k
```

三者的时间窗允许不同。

比较：

| 指标 | S1 | S2 | S3 |
|---|---:|---:|---:|
| event_dt P50/P95 |  |  |  |
| timeout比例 |  |  |  |
| 事件非零率 |  |  |  |
| mask有效率 |  |  |  |
| inv正值比例 |  |  |  |
| inv负值比例 |  |  |  |
| 高风险像素比例 |  |  |  |
| 速度通过率 |  |  |  |
| 角速度通过率 |  |  |  |
| supervise_valid比例 |  |  |  |
| 完整10步序列数量 |  |  |  |
| H5大小 |  |  |  |
| 生成耗时 |  |  |  |

同时记录三者真实事件数：

```text
均应严格等于10000条源事件
```

---

# 九、与EV-TTC官方方案对比

继续复用官方：

```text
exp_filts
ttcef
官方筛选统计
官方TTC与mask
```

官方方案保持：

```text
360×360 signed IIR
固定10 ms窗口
car速度阈值1.3
角速度阈值0.18
```

对比重点：

```text
S1固定10k实际event_dt
S2固定10k实际event_dt
S3固定10k实际event_dt
官方固定10 ms
```

标签数值对比时，不使用最近邻官方样本直接算误差。

需要对抽样step按其各自 `t_start` 重新计算固定10 ms标签，才能比较：

```text
事件窗口标签 vs 固定10 ms标签
```

---

# 十、现有脚本修改要求

重点修改：

```text
build_car_spatial_ablation_from_existing_h5.py
```

增加模式：

```text
--mode paired_window_spatial_ablation
--mode independent_fixed_n_spatial_schemes
```

增加方案参数：

```text
--schemes S1 S2
--event-counts 5000 10000 15000 20000
--index-only
--materialize-n 10000
--resume
```

写入attributes：

```text
experiment_mode
spatial_scheme
event_count_definition
window_source
label_geometry
label_version
```

不要新增一套功能重复的builder。

---

# 十一、输出目录

保留旧目录：

```text
[5]car_spatial_ablation
```

新增修正结果子目录，而不是另起无关项目：

```text
[5]car_spatial_ablation/
├── paired_window_v2/
│   └── N10000_S2_label_v2.h5
├── independent_fixed_n/
│   ├── indices/
│   ├── N10000_S1_360.h5
│   ├── N10000_S2_center256_to128.h5
│   └── reports/
└── reports/
```

不要覆盖旧H5。

---

# 十二、必须生成的报告

```text
旧paired-window数据口径修正报告.md
N10000旧S2与新S2标签对比.md
S1S2全部N独立分片索引统计.md
N10000三种独立固定N方案对比.md
N10000三方案与EVTTC官方对比.md
后续是否扩展其余N的建议.md
```

---

# 十三、最终必须回答

1. 新版S2标签是否恢复局部连续性；
2. 旧S2破碎是否主要由标签和mask池化导致；
3. S1、S2、S3固定10k分别对应多长时间；
4. S1和S2是否都严格累计到10k源事件；
5. S1、S2、S3中哪一个监督质量更好；
6. S2大视野低分辨率是否优于S3局部高分辨率；
7. S1全图是否值得承担更高计算量；
8. 哪些旧结论应继续作废；
9. 是否有必要扩展5k、15k、20k完整H5；
10. 下一步正式训练应优先选择哪一种方案。

完成上述任务后停止，不自动开始训练。
