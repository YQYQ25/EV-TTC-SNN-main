# M3ED固定ROI内1k事件分片与TTC标签生成修正指令

## 一、任务目标

修正当前M3ED调试集的数据生成方式，使“每步1k事件”的定义与MAVLab训练流程一致。

MAVLab的1k事件来自已经裁剪到固定空间区域的事件序列，而不是先从全分辨率事件流取1k，再把其中一部分映射到网络输入区域。

因此，M3ED应采用：

```text
原始事件流
→ 去畸变并映射到360×360
→ 选定固定128×128 ROI
→ 只保留进入该ROI的源事件
→ 按ROI事件流连续取1000个源事件
→ 生成[2,128,128]事件计数图
→ 根据该分片起止时间生成TTC标签
```

当前阶段仍只做数据生成、审计和可视化，不接入正式训练。

---

## 二、先审计当前实现

请检查当前64步debug数据生成代码，明确回答：

1. 当前是否先从全分辨率事件流截取1000条事件，再映射到360×360；
2. 是否执行了与EV-TTC一致的去畸变；
3. 是否采用四邻域双线性分配；
4. 当前每步实际落入360×360有效区域的源事件数是多少；
5. 当前`event_cnt`是整数计数图还是浮点权重图。

在报告中附相关文件路径、函数名和必要源码片段。

---

## 三、重新定义“1k事件”

新的每一步必须由固定ROI内连续出现的1000条源事件构成。

设去畸变并映射后的固定ROI为：

```text
ROI = [x0:x0+128, y0:y0+128]
```

则第k步应满足：

```text
ROI内第1000k条有效源事件
到
ROI内第1000k+999条有效源事件
```

注意：

- 一个源事件经双线性分配后可能贡献到4个像素；
- 这些小数权重仍只对应1条源事件；
- 1k统计的是进入ROI的源事件条数，不是像素贡献数；
- 若事件仅有部分双线性权重进入ROI，第一版应保留该源事件，并记录进入ROI的实际权重；
- 若四个邻域全部落在ROI外，则不计入ROI事件流。

必须保证：

```python
roi_source_event_count == 1000
```

不能用：

```python
event_cnt.sum() == 1000
```

作为判断条件。

---

## 四、空间处理流程

参考：

```text
EV-TTC-main/TTCEF/create_exp.py
```

复用或重写以下逻辑：

```text
相机内参与畸变参数读取
去畸变映射
中央720×720裁剪
2倍下采样到360×360
四邻域双线性权重计算
```

完整空间流程：

```text
M3ED原始1280×720事件
→ 左事件相机去畸变
→ 中央区域x=[280,1000), y=[0,720)
→ 下采样到360×360
→ 在360×360中选定固定128×128 ROI
→ ROI局部坐标归一化到[0,127]
```

例如ROI左上角为`(x0,y0)`，映射后坐标为：

```text
x_roi = x_360 - x0
y_roi = y_360 - y0
```

同一连续序列或同一个训练片段内，ROI位置必须保持不变。

第一版建议固定中心ROI：

```text
x0 = 116
y0 = 116
ROI = [116:244, 116:244]
```

后续再增加随机ROI实验。

---

## 五、双线性分配要求

若去畸变后事件落在：

```text
(x', y') = (x0 + dx, y0 + dy)
```

则分配到四邻域：

```text
w00 = (1-dx)(1-dy)
w10 = dx(1-dy)
w01 = (1-dx)dy
w11 = dxdy
```

正事件累积到正通道，负事件累积到负通道，两个通道均保存非负权重：

```text
event_cnt[0]：正极性权重
event_cnt[1]：负极性权重绝对值
```

输出：

```text
event_cnt.shape = [2,128,128]
event_cnt.dtype = float32
```

本阶段不使用EV-TTC的指数时间衰减。

---

## 六、重新生成时间索引和标签

由于分片边界改变，必须重新计算：

```text
t_start
t_end
dt
T
Omega
depth_start
ttc_start
inverse_ttc_start
valid_ttc_mask
speed_valid
omega_valid
supervise_valid
```

标签仍采用当前确定的起点定义：

```text
深度重投影到t_start
速度由t_start到t_end的相对位姿计算
TTC = Z / (Tz + 1e-5)
inverse TTC = max(0, Tz / Z)
```

标签图还要裁剪到与事件相同的固定128×128 ROI：

```text
event_cnt
depth_start
ttc_start
inverse_ttc_start
valid_ttc_mask
```

五者必须严格使用相同ROI。

---

## 七、重新生成64步debug H5

继续使用：

```text
spot_outdoor_day_skatepark_1
```

生成连续64步，每步包含ROI内1000条源事件。

输出至少包含：

```text
sequence_name
step_index
raw_event_start_idx
raw_event_end_idx
roi_event_rank_start
roi_event_rank_end
t_start
t_end
dt
roi_x0
roi_y0
event_cnt
depth_start
ttc_start
inverse_ttc_start
valid_ttc_mask
T
Omega
speed_valid
omega_valid
supervise_valid
roi_source_event_count
fully_inside_event_count
partially_inside_event_count
total_mapped_weight
positive_source_count
negative_source_count
positive_weight_sum
negative_weight_sum
```

说明：

- `raw_event_start_idx/raw_event_end_idx`记录该1k ROI事件在原始事件流中跨越的索引范围；
- 这个原始索引跨度通常大于1000；
- 相邻step在ROI事件流中必须连续；
- 原始事件流之间也不能重叠或倒序。

---

## 八、必须增加的自动检查

```python
assert roi_source_event_count == 1000
assert roi_event_rank_end - roi_event_rank_start == 1000
assert event_cnt.shape == (2, 128, 128)
assert event_cnt.dtype == np.float32
assert np.all(event_cnt >= 0)
assert np.isfinite(event_cnt).all()
assert t_end > t_start
assert raw_event_end_idx > raw_event_start_idx
```

连续性检查：

```text
当前step的roi_event_rank_start
等于
上一步的roi_event_rank_end
```

空间一致性检查：

```text
event_cnt、depth、TTC、inverse TTC、mask尺寸一致
ROI坐标一致
mask有效位置上的depth和TTC有限
```

统计：

```text
每步dt分布
原始索引跨度分布
映射权重总和
部分落入ROI事件比例
有效标签像素率
T、Omega、Tz分布
TTC和inverse TTC分布
```

---

## 九、与旧debug集对比

请生成一份对比报告，至少比较：

```text
旧方案：全图先取1k，再映射
新方案：固定ROI内取1k
```

对比指标：

```text
每步物理时间dt
每步有效源事件数
event_cnt权重和
10步BPTT覆盖的总物理时间
T和Omega变化幅度
depth/TTC相邻步变化幅度
```

重点判断：

1. 新方案每步dt是否明显增大；
2. 10步内是否出现可观察的运动变化；
3. 新方案是否更接近MAVLab输入统计；
4. 标签与事件是否仍保持空间对齐。

---

## 十、可视化要求

连续选择10步，每步保存：

```text
正事件通道
负事件通道
正负叠加图
depth
TTC
inverse TTC
valid_ttc_mask
```

图中标注：

```text
step index
raw event index range
t_start
t_end
dt
Tz
roi_source_event_count
total_mapped_weight
supervise_valid
ROI坐标
```

同时生成10步时序总览图。

---

## 十一、输出结果

完成后输出：

1. 修改后的源码；
2. 新debug H5路径；
3. `固定ROI内1k事件审计报告.md`；
4. `新旧分片方案对比报告.md`；
5. 可视化目录；
6. `实现说明.md`；
7. 完整运行命令和实际运行结果。

当前阶段不要实现Dataset、模型、损失或正式训练。
