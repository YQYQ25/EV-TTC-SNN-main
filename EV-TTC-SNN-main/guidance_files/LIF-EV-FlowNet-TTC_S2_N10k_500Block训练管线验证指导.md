# LIF-EV-FlowNet-TTC：500-Block训练管线验证指导

## 一、任务目标

在32-Block过拟合实验已经通过后，使用现有S2数据开展：

```text
500个连续Block的训练管线验证
```

本阶段的目的不是获得论文最终精度，而是验证：

```text
正式Dataset/DataLoader
训练集与验证集划分
连续SNN状态管理
训练与验证循环
Checkpoint保存与恢复
Best模型选择
验证指标
推理与可视化
长时间训练稳定性
```

本任务不得自动启动完整S2正式训练，也不得扩展到S1、S3、5k、15k或20k。

---

# 二、固定实验配置

## 2.1 数据

使用：

```text
N10000_S2_center256_to128.h5
```

当前物化数据：

```text
5000 step
= 500个连续10-step Block
```

固定：

```yaml
spatial_scheme: S2_center256_to128
events_per_step: 10000
input_size: [128, 128]
input_channels: 2
input_scale: 0.3
steps_per_block: 10
num_blocks: 500
batch_size: 1
shuffle: false
```

输入模型前仅执行：

```python
events = event_cnt.float() * 0.3
```

禁止额外归一化。

---

## 2.2 模型

使用：

```text
LIF-EV-FlowNet-TTC
```

要求：

```text
MAVLab SpikingMultiResUNetRecurrent主体
输入2通道
输出1通道signed inverse TTC
线性输出
不加载32-Block过拟合权重
从与32-Block实验相同的随机初始化策略重新训练
```

---

## 2.3 Loss

使用已经通过数值对齐的：

```text
EV-TTC masked Charbonnier per-sample loss
```

参数：

```yaml
alpha: 0.45
epsilon: 1.0e-5
smoothness_weight: 0
```

Loss按：

```text
每个样本先按有效像素平均
整个Block再按有效step-sample平均
```

无效监督step仍执行forward，但不参与loss。

---

# 三、训练集与验证集划分

## 3.1 基本划分

按物理时间顺序划分，不随机打乱：

```text
Train：前400个Block
Validation：后100个Block
```

即：

```text
Train：4000 step
Validation：1000 step
```

## 3.2 时间缓冲

检查Block 399与Block 400之间是否存在真实时间连续性。

为减少训练与验证的时间邻近泄漏，建议：

```text
在Train与Validation之间留出至少10个Block作为buffer
```

推荐最终划分：

```text
Train：Block 0-389，共390个Block
Buffer：Block 390-399，共10个Block，不参与训练和验证
Validation：Block 400-499，共100个Block
```

若现有片段中存在timeout或真实时间断裂，优先将划分边界放在断裂处。

输出：

```text
500block_split.csv
```

必须记录：

```text
split
block_index
step_start
step_end
t_start
t_end
reset_required
```

---

# 四、Dataset与DataLoader

## 4.1 Dataset输出

每个样本对应一个完整10-step Block：

```python
{
    "events": [T,2,128,128],
    "target": [T,1,128,128] or [T,128,128],
    "mask": [T,128,128],
    "supervise_valid": [T],
    "reset_required": [T],
    "block_index": scalar,
    "step_indices": [T],
    "t_start": [T],
    "t_end": [T],
}
```

DataLoader后：

```text
events：[B,T,C,H,W]
target：[B,T,1,H,W]或[B,T,H,W]
mask：[B,T,H,W]
supervise_valid：[B,T]
```

当前：

```text
B=1
T=10
C=2
H=W=128
```

## 4.2 顺序要求

训练集：

```text
shuffle=false
按时间顺序读取
```

验证集：

```text
shuffle=false
按时间顺序读取
```

禁止使用随机Block顺序后仍保留跨Block状态。

---

# 五、状态管理

## 5.1 训练阶段

每个Epoch开始：

```python
model.reset_states()
```

Train Block之间：

```text
连续Block：只detach，不reset
timeout或时间断裂：先reset
```

每个Block结束：

```python
model.detach_states()
```

## 5.2 验证阶段

进入Validation前：

```python
model.reset_states()
```

验证过程：

```text
按时间顺序forward
不backward
不optimizer.step
连续Block之间保留状态
每10步detach
timeout或时间断裂处reset
```

禁止Validation继承Train末尾状态。

Validation结束后也应：

```python
model.reset_states()
```

避免影响下一Epoch训练。

---

# 六、训练参数

首轮固定：

```yaml
optimizer: AdamW
learning_rate: 1.0e-3
weight_decay: 0.0
gradient_clip: 100.0
batch_size: 1
max_epochs: 50
input_scale: 0.3
```

来源：

```text
gradient_clip=100.0复用MAVLab train_SNN.yml
```

本阶段不使用：

```text
学习率调度器
早停
数据增强
多尺度Loss
风险加权Loss
符号分类Loss
平滑Loss
预训练权重
```

如50 Epoch仍明显未收敛，可在报告后建议扩展，不自动继续。

---

# 七、训练循环

训练循环保持32-Block实验已验证逻辑：

```python
for epoch in range(max_epochs):
    model.train()
    model.reset_states()

    for block in train_loader:
        if block_has_discontinuity(block):
            model.reset_states()

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
                max_norm=100.0,
            )

            optimizer.step()

        model.detach_states()
```

---

# 八、验证循环

Validation必须复用与训练完全一致的：

```text
input_scale
模型输出
Loss
mask
指标定义
状态规则
```

示意：

```python
model.eval()
model.reset_states()

with torch.no_grad():
    for block in val_loader:
        if block_has_discontinuity(block):
            model.reset_states()

        for t in range(10):
            pred_t = model(block["events"][:, t].float() * 0.3)

            # 计算per-sample loss和各项指标
            # 不backward，不optimizer.step

        model.detach_states()

model.reset_states()
```

---

# 九、训练与验证指标

每个Epoch至少记录：

## 9.1 Loss类

```text
train_charbonnier
val_charbonnier
```

## 9.2 误差类

```text
train_masked_MAE
val_masked_MAE
train_median_AE
val_median_AE
```

## 9.3 Signed TTC相关

```text
正inverse TTC区域MAE
负inverse TTC区域MAE
符号准确率
高风险区域MAE
```

高风险区域先采用：

```text
inverse TTC > 1
```

只作为指标，不进入Loss。

## 9.4 监督统计

```text
总Block数
总step数
forward step数
有效step-sample数
有效像素数
零监督Block数
optimizer更新次数
```

## 9.5 数值稳定性

```text
gradient norm
prediction mean/std/min/max
NaN/Inf计数
```

---

# 十、Best模型选择

主Best模型按：

```text
Validation masked MAE最低
```

保存：

```text
best_val_mae.pt
```

同时保存：

```text
best_val_loss.pt
latest.pt
```

若最佳Loss与最佳MAE不在同一Epoch，两个Checkpoint都保留。

不使用训练Loss选择最终模型。

---

# 十一、Checkpoint恢复测试

必须做一次真实恢复测试：

1. 训练至Epoch 10；
2. 保存 `latest.pt`；
3. 退出程序；
4. 从Checkpoint恢复；
5. 继续训练至Epoch 11或12；
6. 检查优化器状态、Epoch编号和Loss连续性。

Checkpoint必须包含：

```text
model_state_dict
optimizer_state_dict
epoch
global_step
best_val_mae
best_val_loss
模型配置
数据划分
input_scale
随机种子
```

不保存SNN隐藏状态。

恢复后从新Epoch起点开始：

```python
model.reset_states()
```

---

# 十二、可视化

固定选择：

```text
Train锚点：2个
Validation锚点：3个
```

每10 Epoch及Best Epoch保存：

```text
输入正事件
输入负事件
预测signed inverse TTC
GT signed inverse TTC
valid mask
绝对误差图
```

本阶段不再要求所有中间层都大量可视化，但至少在：

```text
Epoch 0
Epoch 10
Epoch 25
Epoch 50
Best Epoch
```

记录主要LIF层：

```text
spike_rate
mem_abs_p99
持续沉默比例
持续高发放比例
```

以确认 `input_scale=0.3` 在更大训练集上仍稳定。

---

# 十三、通过标准

本阶段通过不要求像32-Block那样完全记忆数据。

必须满足：

```text
训练Loss稳定下降
Validation指标能够正常计算
训练与验证无NaN/Inf
Checkpoint保存与恢复成功
Best模型选择正确
Train与Validation状态隔离
DataLoader顺序正确
训练与验证可视化正确
```

泛化表现至少应满足：

```text
Validation预测不是常数图
Validation MAE明显优于随机初始化
Train-Val差距可解释
```

如果Train持续下降而Validation持续恶化，应报告过拟合，但训练管线仍可能判定通过。

---

# 十四、输出目录

建议：

```text
EV-TTC-SNN-main/debug_sets/
lif_evflownet_ttc_s2_n10k_500block_pipeline/
```

包含：

```text
config.yaml
500block_split.csv
train_epoch_metrics.csv
train_block_metrics.csv
val_epoch_metrics.csv
lif_activity_by_epoch.csv
checkpoints/
visualizations/
logs/
```

---

# 十五、必须生成的报告

```text
500Block训练管线验证报告.md
500Block训练验证指标分析.md
500BlockCheckpoint恢复审计.md
500Block训练管线通过或失败判定.md
```

---

# 十六、最终必须回答

1. 500个Block如何划分Train、Buffer和Validation；
2. Dataset/DataLoader输出shape是否正确；
3. 训练和验证是否按时间顺序；
4. Train与Validation之间是否正确reset状态；
5. 无效监督step是否仍执行forward；
6. Loss和MAE是否稳定下降；
7. Validation预测是否明显优于随机初始化；
8. Best模型是否按Validation MAE正确保存；
9. Checkpoint恢复后训练是否连续；
10. scale=0.3是否仍无明显饱和或沉默；
11. 当前训练管线是否满足完整S2正式训练要求；
12. 下一步是否应物化完整S2并进行正式多序列训练。

完成后停止，不自动启动完整S2训练。
