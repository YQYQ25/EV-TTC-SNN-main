# LIF-EV-FlowNet-TTC阶段1至阶段3实现指导

## 一、总目标

在正式32-block过拟合之前，先独立完成并验证：

```text
阶段1：迁移LIF-EV-FlowNet模型
阶段2：实现EV-TTC masked Charbonnier loss
阶段3：完成模型与loss的最小集成测试
```

本任务不得启动32-block过拟合，不得直接开始完整训练。

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
[B,2,128,128]
```

输出：

```text
[B,1,128,128]
```

目标变量：

```text
signed inverse TTC = Tz / Z
```

本阶段只做模型迁移与单元测试，不接正式H5，不实现loss。

---

## 2.2 优先复用内容

从MAVLab原代码中复用：

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
```

除任务输出头外，尽量不改网络主体。

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

即所有最终预测头：

```python
out_channels = 1
```

### 输出激活

最终输出层必须保持线性：

```text
不使用ReLU
不使用Sigmoid
不使用Tanh
```

原因：

```text
inverse TTC允许正值和负值
```

### 模型接口

统一提供：

```python
prediction = model(events_t)
model.reset_states()
model.detach_states()
```

如原代码接口不同，增加兼容封装，不要破坏原LIF模块内部逻辑。

---

## 2.4 128输入测试

使用随机张量：

```python
x = torch.randn(B, 2, 128, 128)
```

检查：

```text
输入shape正确
输出shape为[B,1,128,128]
连续10步forward成功
输出无NaN/Inf
```

---

## 2.5 状态测试

### 状态保留

同一输入连续forward两次：

```python
y1 = model(x)
y2 = model(x)
```

若LIF状态正常保留：

```text
y1和y2允许不同
```

### reset测试

执行：

```python
model.reset_states()
```

再次forward：

```python
y3 = model(x)
```

检查：

```text
reset后状态回到初始状态
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
状态张量不再连接上一段计算图
```

---

## 2.6 10步BPTT测试

构造：

```python
inputs = torch.randn(B, 10, 2, 128, 128)
```

执行：

```python
model.reset_states()

loss = 0
for t in range(10):
    pred = model(inputs[:, t])
    loss = loss + pred.mean()

loss.backward()
```

检查：

```text
backward成功
参数梯度存在
梯度无NaN/Inf
梯度不是全部为0
```

执行：

```python
model.detach_states()
```

确认下一段forward不再连接前10步计算图。

---

## 2.7 360输入兼容测试

S1输入为：

```text
[B,2,360,360]
```

检查网络总下采样倍数。

若360无法被网络总下采样倍数整除：

```text
先padding到合法尺寸
forward
再crop回360×360
```

例如总下采样倍率为16时：

```text
360×360
→ pad到368×368
→ 网络输出
→ crop回360×360
```

禁止直接resize S1。

测试：

```text
输出最终恢复为[B,1,360,360]
无shape mismatch
skip connection尺寸一致
```

---

## 2.8 脉冲活动审计

对连续10步随机输入或真实S2事件样本，记录每个LIF层：

```text
膜电位均值
膜电位标准差
膜电位最大值
脉冲率
全零神经元比例
持续高频发放比例
```

本阶段不决定最终输入缩放，只确认监控接口可以工作。

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

输出报告：

```text
阶段1_LIF-EV-FlowNet-TTC模型迁移报告.md
阶段1_模型单元测试结果.csv
```

未通过时停止，不进入阶段2。

---

# 三、阶段2：实现EV-TTC masked Charbonnier loss

## 3.1 目标

实现支持以下条件的损失：

```text
signed inverse TTC
像素级valid_ttc_mask
样本级supervise_valid
空mask安全处理
无效监督step不参与loss
```

本阶段不连接正式模型训练。

---

## 3.2 推荐接口

实现：

```python
loss, stats = masked_charbonnier_loss(
    prediction,
    target,
    valid_ttc_mask,
    supervise_valid,
)
```

输入：

```text
prediction：[B,1,H,W]或[B,H,W]
target：[B,1,H,W]或[B,H,W]
valid_ttc_mask：[B,H,W]
supervise_valid：[B]
```

---

## 3.3 最终loss mask

构造：

```python
loss_mask = (
    valid_ttc_mask.bool()
    & supervise_valid[:, None, None]
)
```

若prediction有通道维，需要统一shape后再计算。

---

## 3.4 Charbonnier定义

按照EV-TTC原实现读取并复用：

```text
alpha
epsilon
finite处理
归一化方式
```

不要自行假设参数。

形式应与EV-TTC一致：

```text
charbonnier(error)
```

其中：

```text
error = prediction - target
```

signed inverse TTC中的负值正常参与计算。

---

## 3.5 归一化

优先采用：

```text
仅在最终有效像素上做mean
```

即：

```python
loss = charbonnier(error[loss_mask]).mean()
```

同时返回：

```text
valid_pixel_count
valid_sample_count
has_supervision
```

若全batch无有效像素：

```text
has_supervision = False
loss不用于backward
```

不要返回会误触发optimizer.step的伪零损失。

---

## 3.6 人工单元测试

至少完成以下测试。

### 测试A：预测等于标签

```text
prediction == target
```

期望：

```text
loss等于Charbonnier最小值或接近最小值
```

### 测试B：只修改mask外像素

```text
mask内不变
mask外prediction大幅变化
```

期望：

```text
loss不变
```

### 测试C：修改mask内像素

期望：

```text
loss增大
```

### 测试D：supervise_valid=0

将某个样本设为：

```text
supervise_valid=False
```

期望：

```text
该样本完全不影响loss
```

### 测试E：负标签

例如：

```text
target=-0.5
prediction=-0.4
```

期望：

```text
正常计算
不被截断
```

### 测试F：空mask

```text
valid_ttc_mask全0
```

期望：

```text
不产生NaN
has_supervision=False
```

### 测试G：混合batch

batch中同时包含：

```text
有效样本
无效样本
空mask样本
负inverse TTC样本
```

期望：

```text
只对有效监督像素计算
```

---

## 3.7 与EV-TTC原loss数值对齐

使用完全相同的：

```text
prediction
target
mask
参数
```

分别调用：

```text
EV-TTC原loss
新masked Charbonnier
```

统计：

```text
绝对误差
相对误差
```

要求：

```text
仅有浮点误差
```

如果EV-TTC原实现不支持样本级 `supervise_valid`，先用它筛出有效样本，再做数值对照。

---

## 3.8 阶段2通过标准

必须全部满足：

```text
人工测试全部通过
负inverse TTC正常参与
mask外值不影响loss
supervise_valid=0不影响loss
空mask不产生NaN
与EV-TTC原实现数值一致
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

在模型和loss分别通过后，验证最小训练链路：

```text
10步事件输入
→ LIF-EV-FlowNet-TTC
→ masked Charbonnier
→ loss累计
→ backward
→ gradient clipping
→ optimizer.step
→ detach_states
```

本阶段只使用少量人工或真实S2 block，不做32-block过拟合。

---

## 4.2 测试block类型

至少准备三种10步block。

### Block A：10步全部有效

```text
supervise_valid=1
valid_ttc_mask非空
```

预期：

```text
10步全部forward
10步参与loss
执行backward
执行optimizer.step
```

### Block B：部分step有效

例如：

```text
step 0、1、4、7有效
其余无效
```

预期：

```text
10步全部forward
仅4步参与loss
执行一次backward
执行一次optimizer.step
```

### Block C：10步全部无监督

预期：

```text
10步全部forward
0步参与loss
不backward
不optimizer.step
block末尾detach_states
```

---

## 4.3 推荐训练循环结构

```python
model.reset_states()

for block in blocks:
    optimizer.zero_grad(set_to_none=True)

    block_loss_sum = 0.0
    valid_step_count = 0

    for t in range(10):
        pred_t = model(events[:, t])

        step_loss, stats = masked_charbonnier_loss(
            prediction=pred_t,
            target=target[:, t],
            valid_ttc_mask=mask[:, t],
            supervise_valid=supervise_valid[:, t],
        )

        if stats["has_supervision"]:
            block_loss_sum = block_loss_sum + step_loss
            valid_step_count += 1

    if valid_step_count > 0:
        block_loss = block_loss_sum / valid_step_count
        block_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_value,
        )

        optimizer.step()

    model.detach_states()
```

注意：

```text
所有step都必须先forward
不能在forward前跳过无效step
```

---

## 4.4 reset与detach

本阶段分别验证两种模式。

### 连续block模式

相邻block属于连续物理时间：

```text
block之间只detach
不reset
```

### 非连续block模式

block来自随机位置：

```text
每个block开始reset
```

必须在日志中明确当前采用哪一种。

---

## 4.5 梯度检查

对Block A和B检查：

```text
参数梯度存在
梯度无NaN/Inf
gradient clipping实际执行
optimizer.step后参数发生变化
```

对Block C检查：

```text
参数不变化
optimizer.step未调用
```

---

## 4.6 状态检查

记录：

```text
forward_called
loss_included
backward_called
optimizer_step_called
detach_called
reset_called
```

必须满足：

```text
Block A：
forward=10
loss=10
backward=1
optimizer=1

Block B：
forward=10
loss=有效step数
backward=1
optimizer=1

Block C：
forward=10
loss=0
backward=0
optimizer=0
detach=1
```

---

## 4.7 真实S2单block测试

人工block通过后，再读取：

```text
N10000_S2_center256_to128.h5
```

选择1个真实连续10步block。

检查：

```text
event shape
target shape
mask shape
supervise_valid shape
10步forward
loss有限
backward成功
输出无NaN/Inf
```

本阶段不重复训练该block，不观察过拟合。

---

## 4.8 阶段3通过标准

必须全部满足：

```text
三类block行为符合预期
无效step仍forward
零监督block不更新参数
有效block可以backward
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

建议新增或整理：

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

优先复用MAVLab代码，不复制多份同功能模块。

---

# 六、配置记录

配置文件至少保存：

```yaml
model:
  source: MAVLab_LIF_EV_FlowNet
  input_channels: 2
  output_channels: 1
  output_activation: none
  neuron_parameters: MAVLab_original
  surrogate_gradient: MAVLab_original

loss:
  type: masked_charbonnier
  source: EV-TTC
  target: signed_inverse_ttc
  smoothness_weight: 0

sequence:
  steps: 10
  invalid_step_forward: true
  invalid_step_loss: false

state:
  detach_every_steps: 10
  reset_every_block: false
```

具体数值必须从原配置和源码读取，不凭经验填写。

---

# 七、禁止事项

本任务中禁止：

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
```

---

# 八、最终交付

必须提交：

```text
LIF-EV-FlowNet-TTC模型代码
masked Charbonnier loss代码
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
2. 哪些层由2通道输出改为1通道；
3. LIF参数是否与MAVLab一致；
4. reset_states和detach_states是否通过测试；
5. 128和360输入是否均可forward；
6. 新loss是否与EV-TTC数值一致；
7. mask外像素和无效样本是否完全不影响loss；
8. 零监督block是否不会更新参数；
9. 部分有效block是否仍对10步全部forward；
10. 是否已经满足进入32-block过拟合的条件。

完成阶段1至阶段3后停止，等待下一条指令。
