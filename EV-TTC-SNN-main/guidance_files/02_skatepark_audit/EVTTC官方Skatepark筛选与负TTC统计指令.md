# EV-TTC官方脚本生成Skatepark并统计筛选与负TTC比例指令

## 一、任务目标

使用EV-TTC官方仓库的原始数据生成流程，单独处理：

```text
spot_outdoor_day_skatepark_1
```

重点统计：

1. 官方脚本筛选前共有多少候选样本；
2. 经过速度和角速度条件后保留多少样本；
3. 被速度条件、角速度条件分别淘汰多少；
4. 官方生成的TTC中，正TTC、负TTC和零TTC各占多少；
5. 官方mask是否保留负TTC；
6. 官方Skatepark数据的有效样本量是否确实很少；
7. 与当前固定事件数ROI数据的统计差异来自哪里。

本任务以“审计官方实现”为主。不要修改官方筛选条件、TTC公式或mask逻辑。

## 二、使用官方代码

优先直接使用EV-TTC官方仓库中的原始脚本：

```text
EV-TTC-main/TTCEF/create_exp.py
EV-TTC-main/TTCEF/calc_gt.py
EV-TTC-main/merge.py
```

先审计实际调用链，确认：

```text
原始事件表示生成
→ 深度和位姿读取
→ TTC标签生成
→ 样本级速度/角速度筛选
→ merge保存
```

在报告中列出：

```text
实际使用的文件路径
函数名
命令行参数
官方默认阈值
官方mask定义
```

禁止根据记忆重写“等价实现”代替官方脚本。

## 三、核对官方条件

### 3.1 样本级筛选

从源码确认非car序列是否使用：

```text
||T|| > 0.25 m/s
||Omega|| < 0.18 rad/s
```

若源码不同，以源码为准。

### 3.2 TTC计算

确认是否为：

```text
TTC = Z / (Tz + eps)
inverse_TTC = Tz / Z
```

同时确认：

```text
深度取值时刻
运动估计时间窗口
T、Omega所在坐标系
```

### 3.3 像素级mask

检查官方mask是否包含：

```text
TTC < 100
```

以及是否显式包含：

```text
TTC > 0
Tz > 0
depth > 0
finite检查
```

必须附源码片段证明，不能推测。

## 四、生成范围

只处理：

```text
spot_outdoor_day_skatepark_1
```

分两阶段：

### 阶段A：中等规模验证

连续或均匀抽取：

```text
1000～5000个官方候选样本
```

先确认脚本、标签、mask和统计逻辑正确。

### 阶段B：完整序列统计

阶段A无误后，扫描完整序列。

若完整生成过慢，可完整扫描标量运动量和TTC符号统计，不强制保存全部事件表示。

## 五、筛选前后样本统计

对每个候选样本保存：

```text
sample_index
timestamp
T_x
T_y
T_z
||T||
Omega_x
Omega_y
Omega_z
||Omega||
speed_valid
omega_valid
official_sample_valid
```

分类：

```text
A：速度通过，角速度通过
B：仅速度失败
C：仅角速度失败
D：速度和角速度同时失败
```

统计：

```text
候选样本总数
A/B/C/D数量
A/B/C/D比例
官方最终保留样本数
官方保留率
```

额外统计：

```text
||T|| mean/P5/P25/P50/P75/P95
Tz mean/P5/P25/P50/P75/P95
||Omega|| mean/P5/P25/P50/P75/P95
```

## 六、TTC正负比例统计

分别对：

```text
筛选前全部候选样本
通过官方样本级筛选后的样本
```

进行统计。

### 6.1 像素级

在官方实际mask逻辑下统计：

```text
TTC > 0像素数及比例
TTC < 0像素数及比例
TTC == 0像素数及比例
NaN像素数
Inf像素数
inverse_TTC > 0
inverse_TTC < 0
inverse_TTC == 0
```

### 6.2 样本级

统计：

```text
全部TTC为正
全部TTC为负
正负TTC混合
官方mask为空
官方mask非空
Tz > 0
Tz <= 0
```

若TTC只由全局单一 `Tz` 决定，理论上同一样本中的TTC符号应一致；若出现混合，检查实现。

## 七、官方mask与正TTC mask对比

不修改官方输出，额外离线构造：

```text
positive_ttc_mask =
official_mask
AND TTC > 0
AND inverse_TTC > 0
```

比较：

```text
official_mask有效像素率
positive_ttc_mask有效像素率
official_mask空样本比例
positive_ttc_mask空样本比例
```

重点回答：

```text
官方mask是否保留负TTC；
只保留正TTC后还剩多少有效样本。
```

## 八、官方时间口径

确认官方脚本实际运动区间。

若使用固定10 ms，记录：

```text
motion_start
motion_end
motion_dt
```

并确认深度取起始时刻还是结束时刻。

本任务不要改成事件分片起止时间。

## 九、与当前ROI多事件数结果对比

对比已有：

```text
N=5k、10k、15k、20k
```

至少生成：

| 指标 | 官方EV-TTC | 5k | 10k | 15k | 20k |
|---|---:|---:|---:|---:|---:|
| 候选样本数 |  |  |  |  |  |
| 速度通过率 |  |  |  |  |  |
| 角速度通过率 |  |  |  |  |  |
| 官方样本保留率 |  |  |  |  |  |
| Tz>0比例 |  |  |  |  |  |
| 负TTC样本比例 |  |  |  |  |  |
| 官方mask空样本比例 |  |  |  |  |  |
| 正TTC mask空样本比例 |  |  |  |  |  |

分析差异来源：

```text
固定10 ms窗口
事件分片窗口
360×360全图
128×128 ROI
官方是否保留负TTC
官方是否先筛选再保存
```

## 十、可视化抽查

随机抽取以下样本各10个：

```text
官方筛选通过且Tz>0
官方筛选通过但Tz<=0
仅角速度失败
仅速度失败
官方mask非空但正TTC mask为空
```

每个样本保存：

```text
事件表示
depth
TTC
inverse TTC
official_mask
positive_ttc_mask
```

标注：

```text
timestamp
||T||
Tz
||Omega||
speed_valid
omega_valid
official_sample_valid
正TTC比例
负TTC比例
```

## 十一、实现建议

尽量不修改官方源码。

可新增独立审计脚本：

```text
EV-TTC-SNN-main/snn_ttc/tools/
├── run_official_evttc_skatepark_subset.py
├── audit_official_evttc_skatepark.py
└── compare_official_vs_roi_ttc.py
```

若必须增加日志：

- 只增加只读统计和输出；
- 不改变原始计算分支；
- 保存补丁或git diff；
- 报告中明确所有改动。

## 十二、运行要求

运行前检查：

```text
data.h5
depth_gt.h5
pose_gt.h5
输出目录
剩余磁盘空间
```

支持：

```text
--max-samples
--start-index
--full-sequence
--resume
--output-dir
```

阶段A先运行中等规模；确认无误后再决定是否完整扫描。

## 十三、输出文件

至少输出：

1. `EVTTC官方Skatepark生成流程审计.md`
2. `EVTTC官方Skatepark筛选前后统计.md`
3. `EVTTC官方Skatepark负TTC比例统计.md`
4. `官方Mask与正TTCMask对比.md`
5. `官方EVTTC与当前ROI方案对比.md`
6. 候选样本级CSV
7. TTC像素统计CSV
8. 可视化目录
9. 实际运行命令
10. 运行耗时和磁盘占用
11. 所有源码改动或git diff

## 十四、最终必须回答

1. 官方Skatepark候选样本总数是多少；
2. 速度和角速度筛选后保留率是多少；
3. 主要被速度还是角速度淘汰；
4. 官方mask是否保留负TTC；
5. 官方数据中负TTC样本和负TTC像素比例是多少；
6. 只保留正TTC后还剩多少样本；
7. 官方Skatepark有效数据是否确实很少；
8. 当前ROI方案与官方结果是否一致；
9. 差异主要来自筛选条件、时间窗口、ROI还是mask定义；
10. 当前SNN-TTC训练应沿用官方负TTC策略，还是只训练正TTC。
