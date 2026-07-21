# LIF-EV-FlowNet-TTC阶段1至阶段3实现指导（修订版）

## 一、总目标

在正式32-block过拟合之前，依次完成：

```text
阶段1：迁移LIF-EV-FlowNet模型
阶段2：实现EV-TTC masked Charbonnier loss
阶段3：完成模型与loss的最小集成测试
```

本任务不得启动32-block过拟合、500-block训练或完整数据训练。

原则：

```text
先验证模型
再验证loss
最后验证训练链路
```

每个阶段通过后再进入下一阶段。

---

# 二、阶段1：迁移LIF-EV-FlowNet模型

## 2.1 目标

复用MAVLab的LIF-EV-FlowNet主体，构建：

```text
LIF-EV-FlowNet-TTC
```

输入：

```text
[B,2,H,W]
```

输出：

```text
[B,1,H,W]
```

预测目标：

```text
signed inverse TTC = Tz / Z
```

本阶段只做模型迁移与单元测试，不接正式H5，不实现loss。

---

## 2.2 优先复用MAVLab内容

尽量原样复用：

```text
卷积LIF模块
surrogate gradient
encoder-decoder结构
skip connection
状态变量管理
reset_states()
detach_states()
卷积初始化
神经元阈值
膜电位衰减参数
网络通道数
网络层数
梯度裁剪参数
```

除任务输出头和必要的输入尺寸兼容外，不修改网络主体。

---

## 2.3 必须修改

### 输出通道

原光流输出：

```text
2通道：u、v
```

改为：

```text
1通道：signed inverse TTC
```

所有最终预测头统一修改为：

```python
out_channels = 1
```

### 输出激活

最终输出保持线性：

```text
不使用ReLU
不使用Sigmoid
不使用Tanh
```

因为signed inverse TTC允许正值和负值。

### 模型接口

统一提供：

```python
prediction = model(events_t)
model.reset_states()
model.detach_states()
```

如原代码接口不同，增加兼容封装，不破坏原LIF内部实现。

---

## 2.4 128输入测试

使用：

```python
x = torch.randn(B, 2, 128, 128)
```

检查：

```text
输出shape为[B,1,128,128]
连续10步forward成功
输出无NaN/Inf
输出可包含正值和负值
```

---

## 2.5 状态测试

### 状态保留

```python
model.reset_states()
y1 = model(x)
y2 = model(x)
```

若状态正常保留：

```text
y1与y2允许不同
```

### reset测试

```python
model.reset_states()
y3 = model(x)
```

检查：

```text
reset后内部状态回到初始状态
y3应接近第一次forward的y1
```

### detach测试

执行：

```python
model.detach_states()
```

检查：

```text
状态数值不改变
状态张量与上一段计算图断开
```

---

## 2.6 10步BPTT测试

```python
inputs = torch.randn(B, 10, 2, 128, 128)

model.reset_states()

loss = 0.0
for t in range(10):
    pred = model(inputs[:, t])
    loss = loss + pred.mean()

loss.backward()
```

检查：

```text
backward成功
模型参数存在梯度
梯度无NaN/Inf
梯度不是全部为0
```

随后：

```python
model.detach_states()
```

确认下一段forward不再连接前10步计算图。

---

## 2.7 360输入兼容测试

S1输入：

```text
[B,2,360,360]
```

先检查网络总下采样倍率。

如果360不能被总倍率整除：

```text
padding到最近合法尺寸
→ 网络forward
→ 输出crop回360×360
```

例如总倍率为16时：

```text
360×360
→ pad到368×368
→ forward
→ crop回360×360
```

禁止直接resize S1。

检查：

```text
最终输出为[B,1,360,360]
skip connection无shape mismatch
输出无NaN/Inf
```

---

## 2.8 脉冲活动监控接口

为后续输入幅值审计提供每层统计：

```text
膜电位均值
膜电位标准差
膜电位最大值
脉冲率
全零神经元比例
持续高频发放比例
```

本阶段只实现监控，不决定最终输入缩放系数。

---

## 2.9 阶段1通过标准

必须全部满足：

```text
128输入输出shape正确
360输入padding/crop正确
连续10步forward成功
reset_states正确
detach_states正确
10步BPTT成功
梯度无NaN/Inf
输出支持正负值
```

输出：

```text
阶段1_LIF-EV-FlowNet-TTC模型迁移报告.md
阶段1_模型单元测试结果.csv
```

未通过时停止，不进入阶段2。

---

# 三、阶段2：实现EV-TTC masked Charbonnier loss

## 3.1 目标

实现支持以下条件的loss：

```text
signed inverse TTC
像素级valid_ttc_mask
样本级supervise_valid
每个样本独立按有效像素归一化
空mask安全处理
无效样本完全不参与loss
```

本阶段不连接正式模型训练。

---

## 3.2 正确接口

实现：

```python
per_sample_loss, valid_samples, stats = (
    masked_charbonnier_per_sample(
        prediction,
        target,
        valid_ttc_mask,
        supervise_valid,
    )
)
```

输入：

```text
prediction：[B,1,H,W]或[B,H,W]
target：[B,1,H,W]或[B,H,W]
valid_ttc_mask：[B,H,W]
supervise_valid：[B]
```

输出：

```text
per_sample_loss：[B]
valid_samples：[B]，bool
stats：有效像素数、有效样本数等
```

---

## 3.3 最终监督mask

统一shape后构造：

```python
loss_mask = (
    valid_ttc_mask.bool()
    & supervise_valid[:, None, None]
)
```

一个样本只有在：

```text
supervise_valid=True
AND
valid_ttc_mask中至少存在一个有效像素
```

时，才属于 `valid_samples`。

---

## 3.4 Charbonnier定义

严格读取并复用EV-TTC原始实现中的：

```text
alpha
epsilon
finite处理
归一化方式
```

不要自行填写参数。

误差：

```python
error = prediction - target
```

signed inverse TTC中的负值正常参与计算。

TTC回归中：

```text
二阶平滑项权重lambda=0
```

因此第一版loss仅使用masked Charbonnier数据项。

---

## 3.5 每个样本独立归一化

必须按EV-TTC逐样本口径计算：

\[
L_b=
\frac{
\sum_{x,y}M_b(x,y)\rho(e_b(x,y))
}{
\sum_{x,y}M_b(x,y)
}
\]

推荐实现：

```python
if prediction.ndim == 4:
    prediction = prediction[:, 0]
if target.ndim == 4:
    target = target[:, 0]

loss_mask = (
    valid_ttc_mask.bool()
    & supervise_valid[:, None, None]
)

charbonnier_map = charbonnier(prediction - target)

pixel_count = loss_mask.flatten(1).sum(dim=1)
valid_samples = pixel_count > 0

per_sample_loss = torch.zeros(
    prediction.shape[0],
    device=prediction.device,
    dtype=prediction.dtype,
)

loss_sum_per_sample = (
    charbonnier_map * loss_mask.to(charbonnier_map.dtype)
).flatten(1).sum(dim=1)

per_sample_loss[valid_samples] = (
    loss_sum_per_sample[valid_samples]
    / pixel_count[valid_samples].to(prediction.dtype)
)
```

禁止直接对整个batch所有有效像素做总mean，因为这会让有效像素多的样本权重更大。

---

## 3.6 空监督处理

若：

```text
valid_samples.any() == False
```

则：

```text
返回has_supervision=False
不进入backward
不执行optimizer.step
```

不要制造一个可反传的伪零loss。

---

## 3.7 人工单元测试

至少完成：

### 测试A：预测等于标签

```text
prediction == target
```

期望：

```text
loss等于Charbonnier最小值或接近最小值
```

### 测试B：只修改mask外像素

期望：

```text
loss完全不变
```

### 测试C：修改mask内像素

期望：

```text
loss增大
```

### 测试D：supervise_valid=0

期望：

```text
该样本不进入valid_samples
不影响最终loss
```

### 测试E：负标签

例如：

```text
target=-0.5
prediction=-0.4
```

期望：

```text
正常参与回归
不被截断
```

### 测试F：空mask

期望：

```text
不产生NaN
valid_samples=False
```

### 测试G：不同mask面积

构造两个误差分布相同、有效像素数量不同的样本。

期望：

```text
两者per_sample_loss相同或接近
```

用于确认是逐样本归一化，而不是全batch像素平均。

### 测试H：混合batch

batch中同时包含：

```text
有效样本
supervise_valid=0样本
空mask样本
负inverse TTC样本
```

期望：

```text
只返回有效样本的有效loss
```

---

## 3.8 与EV-TTC原实现数值对齐

使用相同：

```text
prediction
target
mask
Charbonnier参数
```

分别调用：

```text
EV-TTC原loss
新loss
```

在逐样本层面比较：

```text
absolute error
relative error
```

要求：

```text
仅存在浮点误差
```

如果EV-TTC原实现没有 `supervise_valid` 参数，先只选择有效样本后进行对照。

---

## 3.9 阶段2通过标准

必须全部满足：

```text
人工测试全部通过
负inverse TTC正常参与
mask外值不影响loss
supervise_valid=0不影响loss
空mask不产生NaN
不同mask面积不会改变样本权重
与EV-TTC原实现逐样本数值一致
```

输出：

```text
阶段2_EVTTC_Masked_Charbonnier实现报告.md
阶段2_Loss单元测试结果.csv
```

未通过时停止，不进入阶段3。

---

# 四、阶段3：模型与loss最小集成测试

## 4.1 目标

验证以下最小训练链路：

```text
10步事件输入
→ LIF-EV-FlowNet-TTC
→ 每步逐样本masked Charbonnier
→ 累计全部有效step-sample
→ 统一backward
→ gradient clipping
→ optimizer.step
→ detach_states
```

本阶段不做32-block过拟合。

---

## 4.2 正确的block loss定义

对batch中第 \(b\) 个样本、第 \(t\) 个step：

\[
L_{b,t}
=
\frac{
\sum_{x,y}M_{b,t}(x,y)
\rho(\hat r_{b,t}-r_{b,t})
}{
\sum_{x,y}M_{b,t}(x,y)
}
\]

定义：

\[
v_{b,t}
=
\text{supervise\_valid}_{b,t}
\land
\left(\sum M_{b,t}>0\right)
\]

整个block：

\[
L_{\mathrm{block}}
=
\frac{
\sum_{t=1}^{T}\sum_{b=1}^{B}
v_{b,t}L_{b,t}
}{
\sum_{t=1}^{T}\sum_{b=1}^{B}v_{b,t}
}
\]

即：

```text
按有效step-sample平均
```

不是：

```text
固定除以10
```

也不是：

```text
先对每个step求batch mean，再按有效step数平均
```

---

## 4.3 正确训练循环

```python
optimizer.zero_grad(set_to_none=True)

loss_sum = None
valid_sample_step_count = 0

for t in range(sequence_steps):
    # 所有step必须forward，不能因监督无效跳过
    pred_t = model(events[:, t])

    per_sample_loss, valid_samples, stats = (
        masked_charbonnier_per_sample(
            prediction=pred_t,
            target=target[:, t],
            valid_ttc_mask=mask[:, t],
            supervise_valid=supervise_valid[:, t],
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

# 无论当前block是否有监督，都要截断上一段计算图
model.detach_states()
```

核心要求：

```text
所有step都执行forward
只累计有效step-sample
零监督block不backward、不optimizer.step
block末尾始终detach_states
```

---

## 4.4 三类测试block

### Block A：全部有效

```text
10步全部supervise_valid=1
全部mask非空
```

预期：

```text
forward调用10次
loss包含B×10个step-sample
backward调用1次
optimizer.step调用1次
detach调用1次
```

### Block B：部分有效

batch内不同样本、不同step具有不同有效性。

预期：

```text
所有step和所有样本都完成forward
只累计valid_samples=True的step-sample
backward调用1次
optimizer.step调用1次
```

### Block C：全部无监督

预期：

```text
forward调用10次
valid_sample_step_count=0
backward不调用
optimizer.step不调用
detach调用1次
```

---

## 4.5 reset与detach

### 连续block模式

相邻block属于同一连续物理时间：

```text
block之间只detach
不reset
```

### 非连续block模式

block来自随机位置或shuffle：

```text
每个block开始reset_states
```

### 必须reset的位置

```text
序列开始
timeout或时间断裂
文件切换
显式跳跃采样
S3 ROI切换
train/validation/test切换
```

禁止每10步机械reset。

---

## 4.6 梯度与参数更新检查

对Block A、B：

```text
梯度存在
梯度无NaN/Inf
grad_norm有记录
gradient clipping执行
optimizer.step后参数发生变化
```

对Block C：

```text
参数不变化
optimizer.step未调用
```

---

## 4.7 运行时审计字段

逐block记录：

```text
block_type
forward_step_count
valid_sample_step_count
backward_called
optimizer_step_called
detach_called
reset_called
grad_norm_before_or_returned
parameter_changed
```

必须满足：

```text
Block A：
forward=10
valid_sample_step_count=B×10
backward=1
optimizer=1

Block B：
forward=10
valid_sample_step_count=真实有效数量
backward=1
optimizer=1

Block C：
forward=10
valid_sample_step_count=0
backward=0
optimizer=0
detach=1
```

---

## 4.8 真实S2单block测试

人工block全部通过后，读取：

```text
N10000_S2_center256_to128.h5
```

选择1个真实连续10步block。

检查：

```text
events：[B,10,2,128,128]
target：[B,10,128,128]或[B,10,1,128,128]
mask：[B,10,128,128]
supervise_valid：[B,10]

10步forward成功
block loss有限
backward成功
梯度有限
optimizer.step成功
detach成功
输出无NaN/Inf
```

本阶段只执行一次或少量更新，不重复训练该block，不观察过拟合。

---

## 4.9 阶段3通过标准

必须全部满足：

```text
三类block行为符合预期
无效step仍forward
loss按有效step-sample平均
零监督block不更新参数
有效block可以完整BPTT
梯度裁剪成功
状态detach/reset正确
真实S2单block集成成功
```

输出：

```text
阶段3_模型与Loss最小集成测试报告.md
阶段3_训练循环运行时审计.csv
```

---

# 五、代码组织建议

建议组织为：

```text
snn_ttc/
├── models/
│   ├── lif_evflownet_ttc.py
│   └── lif_modules.py
├── losses/
│   └── masked_charbonnier.py
├── tests/
│   ├── test_lif_evflownet_ttc.py
│   ├── test_masked_charbonnier.py
│   └── test_model_loss_integration.py
└── configs/
    └── lif_evflownet_ttc_baseline.yaml
```

优先复用MAVLab模块，不复制多份同功能实现。

---

# 六、配置文件要求

至少保存：

```yaml
model:
  source: MAVLab_LIF_EV_FlowNet
  input_channels: 2
  output_channels: 1
  output_activation: none
  neuron_parameters: MAVLab_original
  surrogate_gradient: MAVLab_original

loss:
  type: masked_charbonnier_per_sample
  source: EV-TTC
  target: signed_inverse_ttc
  smoothness_weight: 0
  normalize_pixels_per_sample: true
  average_over_valid_sample_steps: true

sequence:
  steps: 10
  invalid_step_forward: true
  invalid_step_loss: false

state:
  detach_every_steps: 10
  reset_every_block: false
```

具体参数必须从MAVLab与EV-TTC源码、配置中读取，不凭经验填写。

---

# 七、禁止事项

本任务禁止：

```text
启动32-block过拟合
启动500-block训练
启动完整S2训练
修改神经元模型
添加风险加权loss
添加符号分类loss
添加多尺度loss
加载光流预训练权重
使用ReLU截断负inverse TTC
跳过无效监督step的forward
固定除以10
按整个batch全部有效像素直接总平均
```

---

# 八、最终交付

必须提交：

```text
LIF-EV-FlowNet-TTC模型代码
masked Charbonnier per-sample loss代码
模型单元测试
loss单元测试
模型-loss集成测试
baseline配置文件

阶段1_LIF-EV-FlowNet-TTC模型迁移报告.md
阶段2_EVTTC_Masked_Charbonnier实现报告.md
阶段3_模型与Loss最小集成测试报告.md

阶段1_模型单元测试结果.csv
阶段2_Loss单元测试结果.csv
阶段3_训练循环运行时审计.csv
```

---

# 九、最终必须回答

1. MAVLab哪些模块被原样复用；
2. 哪些输出层从2通道改为1通道；
3. LIF参数是否与MAVLab一致；
4. reset_states和detach_states是否通过测试；
5. 128和360输入是否均可forward；
6. 新loss是否与EV-TTC逐样本数值一致；
7. mask外像素和无效样本是否完全不影响loss；
8. 不同mask面积的样本是否保持相同样本权重；
9. 零监督block是否不会更新参数；
10. 部分有效block是否仍对全部10步执行forward；
11. block loss是否严格按有效step-sample平均；
12. 是否已经满足进入32-block过拟合的条件。

完成阶段1至阶段3后停止，等待下一条指令。
