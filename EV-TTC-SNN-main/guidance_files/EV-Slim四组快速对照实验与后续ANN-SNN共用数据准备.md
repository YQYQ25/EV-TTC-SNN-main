# EV-Slim四组快速对照实验与后续ANN/SNN-ANN共用数据准备

## 一、当前阶段目标

当前只完成EV-Slim相关快速诊断，不立即开展完整多序列正式训练，也不立即训练MAVLab的ANN和ANN-SNN混合网络。

需要完成四组EV-Slim实验：

```text
A1：官方6通道IIR + 360×360
A2：官方6通道IIR + 128×128
B1：固定10k事件计数2通道 + 360×360
B2：固定10k事件计数2通道 + 128×128
```

目的：

```text
比较事件表示方式：
官方多时间尺度IIR vs 固定10k事件计数

比较输入分辨率：
360×360 vs 128×128

判断EV-Slim是否能恢复比当前SNN更清晰的dense inverse TTC结构
```

本阶段只做快速探索，不声称跨序列泛化。

---

# 二、实验命名

统一命名：

```text
EVSlim_IIR360
EVSlim_IIR128
EVSlim_Count10k360
EVSlim_Count10k128
```

其中：

```text
EVSlim_IIR360
```

才称为“严格官方EV-Slim基线”。

其余三组称为：

```text
EV-Slim输入表示或分辨率变体
```

---

# 三、两阶段实验策略

## 3.1 阶段A：快速诊断

先只使用当前已有car序列和500-Block时间范围。

统一划分：

```text
Train：Block 0-389，共390个Block
Buffer：Block 390-399，共10个Block
Validation：Block 400-499，共100个Block
```

四组EV-Slim必须使用：

```text
相同原始时间范围
相同Train/Buffer/Validation划分
相同输出时刻
相同标签定义
相同评价样本
```

阶段A只回答：

```text
IIR是否优于Count
360是否优于128
EV-Slim是否能恢复更细致的TTC结构
哪一组值得进一步扩展
```

阶段A不设置Test，不处理额外完整序列。

## 3.2 阶段B：正式实验

只有当阶段A确认某1～2组具有明显优势后，才进行：

```text
Train、Validation、Test采用独立完整序列
```

阶段B不在当前任务内自动启动。

---

# 四、四组实验定义

## 4.1 EVSlim_IIR360

严格按照EV-TTC原论文及官方代码设置：

```text
输入：6通道多时间尺度IIR
尺寸：360×360
网络：官方EV-Slim
输出：1通道inverse TTC
输入和标签生成：优先复用官方代码
训练设置：优先复用官方配置
```

必须从官方代码中确认并记录：

```text
IIR衰减系数
IIR状态更新公式
事件去畸变流程
空间裁剪与缩放流程
输出时间间隔
标签时间戳定义
数据增强
Loss
优化器
学习率
Batch Size
训练Epoch
```

禁止根据论文描述自行近似重写，优先调用或最小修改官方脚本。

---

## 4.2 EVSlim_IIR128

保持与 `EVSlim_IIR360` 相同的：

```text
原始事件
输出时刻
6通道IIR表示
标签
mask
Train/Buffer/Validation划分
EV-Slim主体设计
Loss和优化器
```

仅将空间分辨率改为：

```text
128×128
```

要求：

```text
与360版本保持相同物理视场
```

即将同一个360×360视场的IIR输入、标签和mask一致地下采样或重新映射到128×128。

不能直接使用当前S2中央256视场，否则会同时改变FOV和分辨率。

---

## 4.3 EVSlim_Count10k360

输入：

```text
固定10000条源事件/step
正负极性分离
2通道事件计数
尺寸360×360
```

要求：

```text
每个step在360视场内独立累计10k源事件
记录独立t_start和t_end
标签使用对应时间窗口的位姿、速度、深度和inverse TTC
```

禁止使用S2的10k时间窗后，再在360范围内累计更多事件。

网络使用EV-Slim，但第一层输入通道修改为：

```text
6通道 → 2通道
```

其余主体结构尽量保持不变。

该组不使用SNN的 `input_scale=0.3`。

---

## 4.4 EVSlim_Count10k128

输入：

```text
固定10000条源事件/step
正负极性分离
2通道事件计数
尺寸128×128
```

主实验定义：

```text
保持与Count10k360相同物理视场
360完整视场固定10k
→ 映射或下采样到128×128
```

用于与 `EVSlim_Count10k360` 做纯分辨率比较。

当前S2中央256视场到128版本如需保留，单独命名：

```text
EVSlim_Count10k128_S2FOV
```

不要与纯分辨率128版本混淆。

---

# 五、统一标签与时间定义

阶段A四组必须共享：

```text
相同car序列
相同500-Block时间范围
相同Train/Buffer/Validation划分
相同输出时刻
相同深度来源
相同位姿与速度来源
相同inverse TTC定义
```

signed inverse TTC：

\[
\mathrm{invTTC}=\frac{T_z}{Z}
\]

同时保存：

```text
depth
T
Omega
pose_valid
speed_valid
omega_valid
supervise_valid
valid_ttc_mask
positive_ttc_mask
```

对于严格官方IIR360，如果官方标签定义与当前SNN定义不同：

```text
先保留官方定义完成官方基线
再额外生成统一标签版本用于公平对比
```

---

# 六、监督Mask

至少保存：

```text
event_active_mask
dense_valid_mask
```

定义建议：

```text
event_active_mask
= depth_valid
& reprojection_valid
& event_active
& finite(invTTC)

dense_valid_mask
= depth_valid
& reprojection_valid
& finite(invTTC)
```

阶段A训练优先保持各自原始设置，但统一评价时必须同时报告：

```text
event-active MAE
dense-valid MAE
高风险区域MAE
正inverse TTC区域MAE
负inverse TTC区域MAE
符号准确率
```

---

# 七、阶段A共用索引

建立：

```text
common_debug_index.csv
```

每行至少包含：

```text
sequence_name
split
block_index
sample_id
output_timestamp
raw_event_start_idx
raw_event_end_idx
event_count
t_start
t_end
depth_timestamp
pose_timestamp
reset_required
supervise_valid
```

为不同输入表示建立路径字段：

```text
iir360_sample_path
iir128_sample_path
count10k360_sample_path
count10k128_sample_path
target360_path
target128_path
mask360_path
mask128_path
```

这样后续ANN、ANN-SNN和全SNN可直接复用相同debug数据。

---

# 八、建议数据目录

```text
datasets/
└── m3ed_evttc_debug500/
    ├── metadata/
    │   ├── common_debug_index.csv
    │   ├── debug_split.csv
    │   └── generation_config.yaml
    ├── iir360/
    ├── iir128/
    ├── count10k360/
    ├── count10k128/
    ├── targets360/
    ├── targets128/
    └── masks/
```

优先按序列或输入形式保存为H5，避免生成大量小文件。

---

# 九、EV-Slim网络适配原则

## 9.1 IIR360

```text
不修改官方网络结构
```

## 9.2 IIR128

尽量保持：

```text
同样卷积层数
同样通道数
同样卷积核
同样激活
```

仅为尺寸整除进行必要padding或cropping。

## 9.3 Count10k版本

只修改：

```text
第一层输入通道数：6 → 2
```

其余结构不变。

如果官方第一层存在特殊通道处理逻辑，先审计源码再修改。

---

# 十、训练公平性

四组统一：

```text
相同随机种子
相同Train/Buffer/Validation划分
相同标签
相同评价mask
相同训练Epoch预算
相同Best模型选择标准
相同可视化锚点
```

允许不同：

```text
输入通道数
输入分辨率
事件表示
第一层输入通道适配
```

分别保存：

```text
official_config.yaml
fair_comparison_config.yaml
```

---

# 十一、阶段A评价指标

必须统一报告：

```text
Masked Charbonnier
MAE
Median AE
MRE
高风险区域MAE
正inverse TTC MAE
负inverse TTC MAE
符号准确率
预测std
```

结构相关指标建议增加：

```text
边缘区域MAE
物体内部区域MAE
GT与预测梯度误差
```

效率指标：

```text
参数量
MACs
显存占用
单样本推理时间
训练时间
```

---

# 十二、统一可视化

固定同一批Train和Validation锚点。

每组输出：

```text
输入表示
预测inverse TTC
GT inverse TTC
valid mask
绝对误差图
预测与GT直方图
```

IIR输入显示：

```text
6个时间尺度通道
```

Count10k输入显示：

```text
正事件通道
负事件通道
正负事件总和或差分
```

所有TTC图统一：

```text
相同色图
相同vmin/vmax
显示真实min/max
负值不能被截断
```

---

# 十三、后续ANN和ANN-SNN共用数据准备

当前同步准备阶段A所需的共用debug数据：

```text
2通道Count10k 360×360
2通道Count10k 128×128
6通道IIR 360×360
6通道IIR 128×128
对应统一标签与mask
统一Train/Buffer/Validation索引
```

后续模型：

```text
MAVLab ANN
MAVLab ANN-SNN混合网络
当前全SNN
```

原则：

```text
模型可以不同
数据索引、标签、mask和评价必须一致
```

当前只完成接口审计，不启动训练。

需要生成：

```text
MAVLab_ANN与ANN-SNN接口审计.md
```

至少记录：

```text
输入通道要求
输入尺寸要求
状态管理
输出头尺寸
参数量
Loss接口
是否支持360输入
是否支持6通道IIR
```

---

# 十四、推荐执行顺序

```text
Step 1：审计EV-TTC官方代码与官方输入生成链路
Step 2：基于当前500-Block时间范围生成IIR360
Step 3：训练EVSlim_IIR360
Step 4：生成同FOV的IIR128并训练
Step 5：生成Count10k360和Count10k128
Step 6：训练两个Count版本
Step 7：统一评价四组结果
Step 8：整理ANN、ANN-SNN可复用的debug数据索引
Step 9：审计MAVLab ANN与ANN-SNN接口
Step 10：停止，等待是否进入阶段B
```

---

# 十五、阶段B启动条件

只有在阶段A满足以下条件后，才建议处理独立完整序列：

```text
至少有一组EV-Slim明显优于当前SNN
或者
四组之间出现清晰、可解释的表示与分辨率差异
```

阶段B再进行：

```text
独立Train序列
独立Validation序列
独立Test序列
完整多序列物化与正式训练
```

当前任务禁止自动进入阶段B。

---

# 十六、必须生成的报告

```text
EV-TTC官方代码与输入生成审计.md
EV-Slim四组Debug数据生成报告.md
EV-Slim四组训练配置对照.md
EV-Slim四组500Block结果对比.md
EV-Slim输入表示与分辨率消融结论.md
MAVLab_ANN与ANN-SNN接口审计.md
后续模型共用Debug数据清单.md
```

---

# 十七、最终必须回答

1. `EVSlim_IIR360`是否严格复用了官方实现；
2. IIR128是否保持与360相同物理视场；
3. Count10k360与Count10k128是否分别独立累计10k事件；
4. 四组是否使用相同500-Block时间范围和划分；
5. 360与128的差异是否只包含空间分辨率；
6. IIR与Count的差异是否只包含事件表示；
7. 哪一种输入恢复的局部TTC结构最好；
8. 哪一种配置的高风险区域误差最低；
9. 128输入相对360节省了多少计算量；
10. 是否已准备好ANN、ANN-SNN和全SNN共用debug数据；
11. 阶段A是否足以决定是否进入正式多序列阶段；
12. 是否存在FOV、时间窗口、标签或mask不一致的问题。

完成阶段A后停止，不自动处理独立完整Train、Validation和Test序列。
