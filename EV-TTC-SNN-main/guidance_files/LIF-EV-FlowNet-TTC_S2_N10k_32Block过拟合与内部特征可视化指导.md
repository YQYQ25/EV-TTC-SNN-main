# LIF-EV-FlowNet-TTC：32-Block过拟合与网络内部可视化指导

## 一、任务目标

在已经通过模型、Loss和单Block集成测试的基础上，使用真实S2数据完成：

```text
32个连续Block的过拟合实验
```

本实验只回答：

```text
LIF-EV-FlowNet-TTC能否在当前真实事件输入、signed inverse TTC标签和EV-TTC Loss下学会并记忆小规模训练数据。
```

同时对模型内部数据流进行可视化，展示：

```text
输入
→ 第一层卷积输出
→ 第一层LIF膜电位与脉冲
→ Encoder各级
→ Bottleneck
→ Decoder各级
→ 最终inverse TTC输出
```

每个主要部分选取2～3个代表性特征图，帮助理解网络内部的数据变化。

本任务不进行正式泛化测试，不生成完整训练集，不扩展到S1、S3或其他事件数。

---

# 二、固定实验配置

## 2.1 数据

使用：

```text
N10000_S2_center256_to128.h5
```

配置：

```yaml
spatial_scheme: S2_center256_to128
events_per_step: 10000
input_size: [128, 128]
input_channels: 2
input_scale: 0.3
steps_per_block: 10
num_blocks: 32
total_steps: 320
```

输入模型前只执行：

```python
events = event_cnt.float() * 0.3
```

禁止额外执行：

```text
除以10000
逐帧max归一化
逐通道标准化
clamp
标签缩放
```

---

## 2.2 32个Block的选择

从现有S2 H5中选择32个真实连续Block。

必须满足：

```text
共320个连续step
不跨越timeout
不跨越真实时间断裂
不跨越文件边界
reset_required在片段内部均为False
```

为降低首轮排错难度，优先选择：

```text
supervise_valid=1
valid_ttc_mask非空
```

的连续片段。

保存所选数据索引：

```text
block_index
step_index
t_start
t_end
raw_event_start_idx
raw_event_end_idx
```

输出：

```text
32block_overfit_selection.csv
```

---

## 2.3 Batch Size

第一版固定：

```yaml
batch_size: 1
shuffle: false
```

这里的一个batch为：

```text
[B,T,C,H,W]
=
[1,10,2,128,128]
```

即每次加载一个10步Block。

可视化和日志中必须明确写出：

```text
B=1
T=10
C=2
H=128
W=128
```

虽然当前 `batch_size=1`，所有实现仍必须保留batch维，不允许通过 `squeeze()` 随意删除batch维。

本阶段不使用batch_size大于1，避免不同时间序列的SNN状态管理产生额外变量。

---

# 三、模型与Loss

## 3.1 模型

使用已经通过测试的：

```text
LIF-EV-FlowNet-TTC
```

要求：

```text
MAVLab SpikingMultiResUNetRecurrent主体
输入2通道
输出1通道
输出激活=None
随机初始化
不加载光流预训练权重
```

## 3.2 Loss

使用已经完成数值对齐的：

```text
EV-TTC masked Charbonnier per-sample loss
```

参数：

```yaml
alpha: 0.45
epsilon: 1.0e-5
smoothness_weight: 0
```

监督目标：

```text
signed inverse TTC = Tz / Z
```

整个Block的Loss按：

```text
有效step-sample平均
```

不能固定除以10，也不能对整个batch的全部有效像素直接做一次总平均。

---

# 四、训练状态管理

## 4.1 Epoch开始

每个Epoch开始时：

```python
model.reset_states()
```

原因：

```text
新Epoch再次从32-Block片段起点开始，不能继承上一个Epoch末尾状态。
```

## 4.2 Block之间

32个Block按真实时间顺序输入。

每个Block完成10步BPTT后：

```python
model.detach_states()
```

但连续Block之间禁止：

```python
model.reset_states()
```

即：

```text
Block 0开始：reset
Block 0结束：detach
Block 1开始：不reset
Block 1结束：detach
...
Block 31结束：detach
```

## 4.3 无监督Step

无论监督是否有效：

```text
所有Step都必须执行forward
```

无效监督Step：

```text
不参与Loss
但继续更新SNN状态
```

若整个Block无监督：

```text
完成10步forward
不backward
不optimizer.step
正常detach_states
```

---

# 五、优化参数

优先读取并记录MAVLab原训练配置中的：

```text
optimizer参数
weight decay
gradient clip
权重初始化
LIF参数
surrogate gradient参数
```

首轮建议：

```yaml
optimizer: AdamW
learning_rate: 1.0e-3
batch_size: 1
max_epochs: 300
checkpoint_interval: 10
visualization_interval: 10
```

梯度裁剪必须使用MAVLab原值；如果原配置无法确认，停止并在报告中说明，不能静默使用自定义值。

不使用：

```text
数据增强
学习率调度器
早停
多尺度Loss
风险加权Loss
符号分类Loss
空间平滑Loss
脉冲率正则
```

---

# 六、训练循环

```python
for epoch in range(max_epochs):
    model.train()
    model.reset_states()

    for block in ordered_blocks:
        optimizer.zero_grad(set_to_none=True)

        loss_sum = None
        valid_sample_step_count = 0

        for t in range(10):
            events_t = block["events"][:, t].float() * 0.3

            pred_t = model(events_t)

            per_sample_loss, valid_samples, stats = (
                masked_charbonnier_per_sample(
                    prediction=pred_t,
                    target=block["target"][:, t],
                    valid_ttc_mask=block["mask"][:, t],
                    supervise_valid=block["supervise_valid"][:, t],
                )
            )

            if valid_samples.any():
                current_sum = per_sample_loss[valid_samples].sum()

                loss_sum = (
                    current_sum
                    if loss_sum is None
                    else loss_sum + current_sum
                )

                valid_sample_step_count += int(
                    valid_samples.sum().item()
                )

        if valid_sample_step_count > 0:
            block_loss = loss_sum / valid_sample_step_count
            block_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_value,
            )

            optimizer.step()

        model.detach_states()
```

---

# 七、内部特征可视化

## 7.1 固定可视化样本

从32个Block中固定选择3个锚点：

```text
锚点A：Block 0，Step 0
锚点B：Block 15或16，Step 5
锚点C：Block 31，Step 9
```

锚点必须覆盖：

```text
片段起点
片段中部
片段末端
```

整个训练过程中始终使用相同：

```text
block
step
batch_index
feature channel
```

禁止每个Epoch重新挑选“最好看”的通道。

---

## 7.2 可视化Epoch

至少在以下时刻保存内部特征：

```text
Epoch 0：训练前
Epoch 1
Epoch 5
Epoch 10
Epoch 20
Epoch 50
Epoch 100
Epoch 200
Epoch 300或最终Epoch
Best Epoch
```

---

## 7.3 输入可视化

对每个锚点显示完整输入tensor信息：

```text
shape = [B,2,128,128]
batch_size = B
batch_index = 0
input_scale = 0.3
block_index
step_index
```

至少输出以下3张图：

```text
正事件通道
负事件通道
正负事件总和或signed difference
```

图片标题必须包含：

```text
B、T、C、H、W
当前batch index
当前step
原始事件权重总和
缩放后权重总和
```

---

## 7.4 第一层可视化

第一层至少拆分为：

```text
第一层卷积/输入电流输出
第一层LIF膜电位
第一层LIF脉冲
```

每部分选择2～3个固定代表通道。

通道选择方法：

```text
在Epoch 0上确定一次
之后固定不变
```

推荐从每层选择：

```text
通道1：空间标准差最高
通道2：活动程度接近该层中位数
通道3：低活动但非全零
```

输出：

```text
first_conv_current
enc1_membrane
enc1_spikes
```

---

## 7.5 Encoder可视化

对每个Encoder Stage：

```text
enc1
enc2
enc3
enc4
```

分别保存：

```text
膜电位2～3个固定通道
脉冲2～3个固定通道
```

图片必须标明：

```text
tensor shape
feature channel index
spike rate
membrane mean/std
空间尺寸
```

如果Encoder同时存在卷积电流输出，可对 `enc1` 和 `enc4` 额外保存2～3个卷积电流通道，不要求每一级都重复保存。

---

## 7.6 Bottleneck可视化

对：

```text
bottleneck_res1_lif
bottleneck_res2_lif
```

分别保存：

```text
膜电位2～3个固定通道
脉冲2～3个固定通道
```

重点观察：

```text
是否持续高发放
空间结构是否仍可辨认
训练后是否出现全零或全一
```

---

## 7.7 Decoder可视化

对：

```text
dec1
dec2
dec3
dec4
```

分别保存：

```text
膜电位2～3个固定通道
脉冲2～3个固定通道
```

由于Decoder初始脉冲率较低，应额外记录：

```text
非零脉冲像素数
spike_rate
持续沉默比例
```

若某固定通道在Epoch 0为全零，不要每个Epoch随意换通道；允许在Epoch 0选择非零且具有代表性的通道后固定。

---

## 7.8 输出层可视化

每个锚点至少保存：

```text
预测signed inverse TTC
GT signed inverse TTC
valid_ttc_mask
mask后的预测
绝对误差图
```

建议再保存：

```text
预测与GT并排图
预测、GT和误差的直方图
```

图片标题标明：

```text
Epoch
Block
Step
masked MAE
Charbonnier loss
prediction mean/std/min/max
GT mean/std/min/max
```

---

## 7.9 每部分2～3张图的口径

```text
输入：正事件、负事件、事件总和，共3张
卷积/膜电位/脉冲：每个tensor选2～3个固定通道
输出层：预测、GT、误差，至少3张
```

为了避免生成数百个零散文件，可以将同一层的3个通道组合为一张三列Figure，但必须同时保存：

```text
原始数组.npz
通道编号
原始数值范围
```

---

# 八、可视化一致性要求

## 8.1 固定通道

将各层选中的通道写入：

```text
selected_feature_channels.json
```

训练前确定后，不再改变。

## 8.2 固定颜色范围

不能对每个Epoch单独自动归一化。

要求：

1. 训练时保存原始feature数组；
2. 训练结束后统计各阶段统一稳健范围；
3. 使用相同层、相同通道、相同vmin/vmax重新生成跨Epoch对比图。

可使用：

```text
1%～99% percentile
```

确定连续特征的统一范围。

脉冲图固定：

```text
vmin=0
vmax=1
```

## 8.3 上采样方式

仅用于显示时：

```text
脉冲图：nearest
膜电位/连续特征：bilinear
```

不得修改保存的原始特征。

---

# 九、训练监控

每个Epoch记录：

```text
平均Block Loss
masked MAE
median absolute error
有效step-sample数
有效像素数
gradient norm
参数更新次数
prediction mean/std/min/max
NaN/Inf
```

每个主要LIF层记录：

```text
平均spike_rate
持续沉默比例
持续高发放比例
mem_mean
mem_std
mem_abs_p99
```

重点检查scale=0.3在训练后是否仍合理：

```text
Encoder/Bottleneck是否进一步高发放
Decoder是否重新沉默
膜电位是否发散
```

---

# 十、Checkpoint与恢复

保存：

```text
latest.pt
best_loss.pt
epoch_000.pt
epoch_010.pt
epoch_020.pt
epoch_050.pt
epoch_100.pt
epoch_200.pt
epoch_300.pt
```

Checkpoint必须包含：

```text
model_state_dict
optimizer_state_dict
epoch
global_step
input_scale
模型配置
Loss配置
所选32个Block索引
随机种子
```

不保存跨Epoch的SNN隐藏状态。恢复新Epoch训练时重新：

```python
model.reset_states()
```

---

# 十一、过拟合通过标准

### 数值标准

```text
Loss显著下降
masked MAE显著下降
输出无NaN/Inf
梯度有限
参数持续更新
```

建议参考：

```text
最终Loss相比初始下降80%以上
```

该比例不是唯一硬门槛。

### 图像标准

固定锚点上应看到：

```text
预测不再是近常数图
道路纵向风险梯度逐渐出现
车辆、树木或路侧障碍轮廓逐渐出现
高inverse TTC区域位置与GT逐渐一致
误差图明显减弱
```

### SNN活动标准

```text
Decoder不完全沉默
Encoder/Bottleneck不大面积饱和
膜电位不持续发散
```

---

# 十二、异常排查顺序

若不能过拟合，严格按：

```text
1. 固定锚点GT、mask和事件输入是否对齐
2. Loss mask和supervise_valid是否正确
3. input_scale=0.3是否在训练后导致饱和或沉默
4. 输出标签量级与预测量级是否严重不匹配
5. 10步Loss累计和有效step-sample平均是否正确
6. reset_states/detach_states是否正确
7. 梯度是否到达Encoder和Decoder
8. 学习率与梯度裁剪
9. 最后才考虑模型结构
```

不要首先添加复杂Loss或修改神经元。

---

# 十三、输出目录

```text
EV-TTC-SNN-main/debug_sets/
lif_evflownet_ttc_s2_n10k_overfit32/
```

包含：

```text
config.yaml
32block_overfit_selection.csv
train_epoch_metrics.csv
train_block_metrics.csv
lif_activity_by_epoch.csv
selected_feature_channels.json
checkpoints/
visualizations/
features_npz/
logs/
```

---

# 十四、必须生成的报告

```text
32Block过拟合训练报告.md
32Block过拟合内部特征可视化说明.md
32Block过拟合失败或通过判定.md
```

主报告必须明确：

```text
训练是否成功
最终Loss与MAE下降幅度
最佳Epoch
模型是否记住32个Block
scale=0.3训练后是否仍合理
内部特征从输入到输出如何变化
```

---

# 十五、最终必须回答

1. 32个连续Block是否正确选择；
2. Batch tensor完整shape是什么；
3. input_scale是否只作用于事件输入；
4. 训练Loss是否明显下降；
5. 模型是否能记忆32个Block；
6. 第一层如何将正负事件转换为特征；
7. Encoder各级空间尺寸和特征变化是什么；
8. Bottleneck是否保留有效空间结构；
9. Decoder是否从低分辨率特征恢复TTC结构；
10. 输出层预测是否逐渐接近GT；
11. scale=0.3在训练过程中是否导致饱和或沉默；
12. 是否满足进入500-Block训练管线验证的条件。

完成后停止，不自动启动500-Block训练。
