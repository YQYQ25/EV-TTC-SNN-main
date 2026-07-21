# Car三种空间方案既有文件复用与修改审计指令

## 一、任务目的

不要新开一套实验，也不要立即重新生成数据。

先审计上一次“Car多事件数三种空间方案与EV-TTC官方对比”实验已经产生的全部文件，明确：

1. 哪些文件可以直接复用；
2. 哪些文件只能作为“同时间窗空间对齐辅助实验”复用；
3. 哪些文件需要修改代码后重新生成部分字段；
4. 哪些结果已经因实验设计错误而失效；
5. 应在现有脚本和目录上如何最小修改，而不是另起炉灶。

本次只输出审计报告和修改计划，不执行大规模重生成。

## 二、确认上次实验的实际数据链路

需要明确上次实验是否采用了以下设计：

```text
先使用S3随机128 ROI内累计N事件得到[t_start,t_end]
再用同一[t_start,t_end]生成S1、S2、S3
```

若是，则说明：

- S3仍是实际“ROI内固定N事件”方案；
- S1不是“360全图内固定N事件”，只是“S3时间窗内的360全图事件”；
- S2不是“中央大视野内固定N事件”，只是“S3时间窗内的S1下采样版本”；
- S1/S2/S3事件数并不相等；
- 该数据只能作为“同时间窗空间消融辅助集”，不能作为三种固定N主实验。

必须用源码和H5字段验证，不能只根据指导文件推断。

## 三、扫描并列出现有文件

递归扫描上次实验目录，包括但不限于：

```text
EV-TTC-SNN-main/debug_sets/car_spatial_ablation/
EV-TTC-SNN-main/analysis/
EV-TTC-SNN-main/snn_ttc/tools/
```

以及实际日志、CSV、H5、可视化和Markdown报告目录。

输出完整文件清单：

```text
文件路径
文件类型
文件大小
修改时间
生成脚本
输入依赖
是否完整
是否可断点续跑
```

重点寻找：

```text
N5000_spatial_compare.h5
N10000_spatial_compare.h5
N15000_spatial_compare.h5
N20000_spatial_compare.h5

已有5k/10k/15k/20k原始S3 H5
官方EV-TTC exp_filts
官方EV-TTC ttcef
step级CSV
block级CSV
日志
可视化
空间对齐审计
事件统计
标签统计
计算量统计
综合报告
```

## 四、检查每个H5的真实字段和数据来源

对每个H5输出：

```text
dataset名称
shape
dtype
压缩方式
样本数
block数
是否包含raw event索引
是否包含t_start/t_end
是否包含ROI坐标
是否包含T/Omega
是否包含depth
是否包含signed inverse TTC
是否包含mask
是否包含positive mask
```

并追踪各方案来源。

### S1

确认：

```text
event_cnt_360是否由S3分片的同一时间窗重编码
T/Omega是否沿用S3
depth/inverse TTC/mask是否在360几何上重新计算
```

### S2

确认：

```text
event是否由S1做2×2 sum pooling
depth是否由S1做valid mean pooling
inverse TTC是否由S1做valid mean pooling
mask是否由S1做2×2 any pooling
```

### S3

确认：

```text
是否由当前128×128 ROI内累计恰好N条源事件
是否按自身[t_start,t_end]计算T/Omega和TTC
```

## 五、文件分类

将全部文件分成四类。

### A类：可以直接复用

预期可能包括：

```text
原始M3ED data/depth/pose
EV-TTC官方exp_filts与ttcef
相机标定
去畸变LUT
原始S3固定N H5
原始事件分片索引
ROI轨迹
位姿插值缓存
已有官方EV-TTC审计结果
```

必须实际确认。

### B类：可作为辅助“同时间窗空间对齐集”复用

预期可能包括：

```text
上次spatial_compare H5中的S1
上次spatial_compare H5中的S3
S1/S3配对可视化
同时间窗下的事件密度和视野比较
```

这些不能作为“三种方案固定N主实验”，但可保留为：

```text
相同[t_start,t_end]下的纯空间消融辅助数据
```

### C类：需要局部重算或修改

预期包括：

```text
S2 depth
S2 signed inverse TTC
S2 mask
S2 positive mask
S2相关统计与可视化
```

原因：

```text
当前S2标签由S1标签池化得到
当前S2 mask由S1最终mask做any pooling得到
```

即使保留为同时间窗辅助集，S2标签也应按S2目标几何重新投影depth并重新计算。

### D类：主实验口径下失效

预期包括：

```text
将S1称为“360全图固定N”的结论
将S2称为“中央大视野固定N”的结论
S1/S2/S3三者事件数相等的假设
基于上述假设形成的最终推荐排序
```

明确列出哪些Markdown、CSV和图表结论必须作废或重写。

## 六、判断哪些现有数据能支持正确主实验

正确的固定N主实验应为：

```text
S1：360×360全视野内独立累计N条事件
S2：对应中央约256×256原始视野内独立累计N条事件
S3：随机128×128 ROI内独立累计N条事件
```

三者应分别得到：

```text
各自的raw_event_start_idx/raw_event_end_idx
各自的t_start/t_end
各自的event_dt
各自的T/Omega
各自的depth/TTC/mask
```

检查：

1. 是否已经存在S1独立固定N索引；
2. 是否已经存在S2独立固定N索引；
3. 是否只有S3独立固定N索引；
4. 能否利用已有全序列ROI扫描结果或中间缓存快速构建S1/S2索引；
5. 是否必须重新扫描原始事件流；
6. 若必须重新扫描，哪些缓存可显著减少耗时。

## 七、优先修改现有脚本，不新增平行实现

重点检查并报告以下现有脚本的真实路径和功能：

```text
build_car_spatial_ablation_from_existing_h5.py
downsample_360_to_center128.py
audit_car_spatial_alignment.py
compare_car_spatial_schemes.py
estimate_snn_spatial_compute.py
visualize_car_spatial_ablation.py
```

若实际名称不同，列出真实文件。

对每个脚本说明：

```text
可以原样复用
需要修改
应废弃
```

修改原则：

- 不重新写一套重复代码；
- 优先扩展现有builder支持 `mode=paired_window` 与 `mode=independent_fixed_n`；
- 复用已有几何、投影、H5写入、checkpoint和日志代码；
- 将错误的S2标签池化逻辑替换为S2目标几何下重新生成；
- 保留旧结果目录，不覆盖，增加版本或状态标记。

## 八、建议重命名数据口径

为避免继续混淆，将上次已有数据标记为：

```text
paired_window_spatial_ablation
```

未来正确主实验标记为：

```text
independent_fixed_n_spatial_schemes
```

在H5 attributes中至少加入：

```text
experiment_mode
event_count_definition
spatial_scheme
window_source
label_geometry
label_version
```

旧S1示例：

```text
experiment_mode = paired_window_spatial_ablation
window_source = S3_random128
event_count_definition = variable_in_S1
```

禁止继续把旧S1标为“360 fixed N”。

## 九、修改成本估计

对正确主实验分别估算：

```text
S1独立固定N重新扫描耗时
S2独立固定N重新扫描耗时
S3是否无需重算
S2标签修正耗时
新增H5磁盘占用
可复用缓存节省的耗时
```

按N分别估算：

```text
5k
10k
15k
20k
```

提出最小执行顺序，例如：

```text
先只修正N=10k
验证S1/S2/S3生成逻辑
再决定是否扩展到其余N
```

本次只报告，不实际执行。

## 十、必须输出的报告

生成：

```text
Car三种空间方案既有文件复用审计.md
Car三种空间方案H5字段与数据来源.md
Car三种空间方案脚本修改清单.md
Car三种空间方案失效结果清单.md
Car三种空间方案最小修正执行计划.md
```

主报告至少包含：

| 文件/结果 | 当前含义 | 是否可复用 | 复用用途 | 是否需修改 | 具体修改 |
|---|---|---|---|---|---|

脚本表：

| 脚本 | 当前功能 | 问题 | 修改方式 | 是否保留 |
|---|---|---|---|---|

## 十一、最终必须明确回答

1. 当前S1、S2、S3的事件时间窗分别来自哪里；
2. 当前S1和S2是否真的各自包含N条事件；
3. 当前S3是否可以直接作为正确固定N方案复用；
4. 当前S1可以保留做什么；
5. 当前S2哪些字段必须重算；
6. 哪些统计报告和推荐结论已经失效；
7. 哪些现有脚本应直接修改；
8. 是否必须重新扫描原始74.88 GiB事件文件；
9. 可复用哪些索引、缓存、几何和标签；
10. 最小修改路径是什么。

本次完成报告后停止，不开始重新生成。
