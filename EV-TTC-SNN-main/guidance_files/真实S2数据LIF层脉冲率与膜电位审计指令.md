# 真实S2数据LIF层脉冲率与膜电位审计指令

## 一、任务目标

在进行32-block过拟合之前，先使用真实S2数据检查：

```text
各LIF层是否正常发放
膜电位是否明显饱和或持续累积
当前10k事件输入是否需要缩放
模型输出是否退化为常数
```

本任务只做forward审计：

```text
不计算loss
不执行backward
不更新参数
不启动过拟合训练
```

---

## 二、使用数据

使用：

```text
EV-TTC-SNN-main/debug_sets/[5]car_spatial_ablation/
independent_fixed_n/N10000_S2_center256_to128.h5
```

输入字段：

```text
event_cnt
```

输入尺寸：

```text
[2,128,128]
```

选择：

```text
20个连续10步block
= 200个连续step
```

要求：

- 从真实连续时间片段中选择；
- 不随机打乱；
- 不跨越timeout或真实时间断裂；
- 第一个block开始时执行一次 `reset_states()`；
- 每10步结束仅执行 `detach_states()`；
- 连续block之间不reset。

---

## 三、模型与配置

使用已经通过阶段1测试的：

```text
LIF-EV-FlowNet-TTC
```

要求：

```text
复用MAVLab原LIF参数
输出通道为1
输出激活为None
模型处于eval模式
torch.no_grad()
```

不要加载训练后权重，使用当前基线初始化。

---

## 四、输入方案

至少审计以下两种输入：

### A：原始事件计数

```python
x = event_cnt.float()
```

### B：统一缩放0.1

```python
x = event_cnt.float() * 0.1
```

目的不是默认采用0.1，而是判断原始10k事件输入是否相对MAVLab的1k事件输入造成过强神经活动。

除缩放系数外，两次forward必须使用：

```text
相同模型初始化
相同200个step
相同状态管理
```

建议固定随机种子，并为A、B分别重新实例化同一初始权重模型，避免前一次状态影响后一次结果。

---

## 五、需要统计的层

对所有LIF层逐层统计，至少覆盖：

```text
encoder各LIF层
bottleneck/recurrent LIF层
decoder各LIF层
```

输出头若为非脉冲实值卷积，则不统计脉冲率，但要统计其输出分布。

为每层记录真实层名，例如：

```text
enc1_lif
enc2_lif
...
dec1_lif
```

不要只使用编号而无法对应源码。

---

## 六、每层统计指标

对每个step、每个LIF层记录：

### 脉冲统计

```text
spike_rate
zero_spike_ratio
high_firing_ratio
```

定义建议：

\[
\text{spike rate}
=
\frac{\text{spike元素中非零数量}}
{\text{spike元素总数}}
\]

```text
zero_spike_ratio：
该step内从未发放的神经元比例

high_firing_ratio：
当前step内接近持续发放的神经元比例
```

若单step每个神经元只有0/1脉冲，则：

```text
high_firing_ratio可定义为spike=1的神经元比例
```

同时应增加跨200步统计：

```text
每个神经元的时间平均发放率
持续高发放神经元比例
持续沉默神经元比例
```

### 膜电位统计

```text
mem_mean
mem_std
mem_min
mem_max
mem_abs_p95
mem_abs_p99
finite_ratio
```

### 输出统计

对最终signed inverse TTC预测记录：

```text
output_mean
output_std
output_min
output_max
output_abs_p95
positive_ratio
negative_ratio
finite_ratio
```

---

## 七、时间维度检查

除总平均外，必须绘制或保存：

```text
step 1至200的每层spike_rate曲线
step 1至200的每层mem_mean曲线
step 1至200的每层mem_abs_p99曲线
最终输出mean/std随时间变化
```

重点检查：

```text
膜电位是否随时间持续单调增大
脉冲率是否逐步逼近0或1
detach后状态数值是否连续
block边界是否出现非预期跳变
```

在每个10步block边界画竖线，确认：

```text
detach_states不会清空状态
```

---

## 八、reset与detach验证

在正式200步审计前，额外做一个小对照：

### 连续状态模式

```text
第1步前reset
每10步detach
block之间不reset
```

### 每block reset模式

```text
每10步block开始reset
```

只需比较前5个block，统计：

```text
各层平均spike_rate
mem_mean/mem_std
最终输出差异
```

目的：

```text
确认连续状态与机械reset确实产生不同结果
验证当前状态管理接口真实生效
```

该对照只做审计，不作为正式训练策略。

---

## 九、异常判据

不要使用单一硬阈值直接决定输入缩放，但至少检查以下异常：

### 明显沉默

```text
多数主要LIF层spike_rate长期接近0
输出std极小
预测接近常数
```

### 明显饱和

```text
多数主要LIF层spike_rate长期接近1
大量神经元持续每步发放
膜电位绝对值持续升高
```

### 数值异常

```text
出现NaN/Inf
mem_abs_p99异常增大
输出范围持续发散
```

### 状态异常

```text
detach后状态被清零
不reset时相邻block输出完全不连续
reset后输出无法回到初始响应
```

---

## 十、输入缩放判断

最终比较A与B：

| 指标 | 原始计数 | ×0.1 |
|---|---:|---:|
| 各层平均spike_rate |  |  |
| 沉默层数量 |  |  |
| 饱和层数量 |  |  |
| mem_abs_p99 |  |  |
| 输出std |  |  |
| 输出正负比例 |  |  |
| 是否出现NaN/Inf |  |  |

判断原则：

```text
若原始计数下各层活动正常：
保持scale=1.0

若原始计数明显饱和，而×0.1恢复正常：
后续再细化测试0.2、0.5等尺度

若原始计数明显沉默：
不要继续缩小输入，优先检查阈值、权重初始化和事件幅值
```

本任务不要自动确定最终缩放系数，只给出建议。

---

## 十一、实现建议

优先复用阶段1已经实现的监控接口。

建议新增：

```text
snn_ttc/tools/audit_real_s2_lif_activity.py
```

支持参数：

```text
--h5 <S2 H5路径>
--num-blocks 20
--steps-per-block 10
--input-scales 1.0 0.1
--seed 42
--device cuda
--resume
```

要求：

- 使用forward hook或模型已有状态访问接口；
- 不修改模型数值逻辑；
- 不在hook中detach原始状态导致模型行为变化；
- 审计结束后移除所有hook。

---

## 十二、输出文件

输出目录建议：

```text
EV-TTC-SNN-main/debug_sets/lif_evflownet_ttc_activity_audit/
```

至少生成：

```text
真实S2_LIF活动审计报告.md
真实S2_LIF逐层统计.csv
真实S2_LIF逐step统计.csv
真实S2_输出统计.csv
真实S2_连续状态与每block_reset对比.csv
```

图像至少包括：

```text
各层平均spike_rate对比图
各层mem_abs_p99对比图
spike_rate随step变化图
mem_mean随step变化图
mem_abs_p99随step变化图
输出mean/std随step变化图
```

原始计数与×0.1分别输出，或在同一图中明确区分。

---

## 十三、最终必须回答

1. 原始10k事件计数下，各LIF层是否明显沉默；
2. 原始10k事件计数下，各LIF层是否明显饱和；
3. 膜电位是否随200步持续异常累积；
4. 每10步detach后状态是否正确保留；
5. 每block reset与连续状态模式是否产生预期差异；
6. 最终输出是否接近常数；
7. 是否存在NaN、Inf或异常极值；
8. 原始计数和×0.1哪一个更合理；
9. 是否需要继续测试0.2、0.5等缩放系数；
10. 当前是否满足进入32-block过拟合的条件。

完成审计后停止，不启动32-block过拟合。
