# car单序列严格官方EV-TTC复现指导（单4090与Epoch扩展版）

## 一、任务目标

在原始序列：

```text
car_urban_night_rittenhouse
```

上，严格复用EV-TTC官方的数据生成、标签、Mask、网络与训练链路。

本任务只做：

```text
单序列、严格官方数据生成与训练链路复现
```

由于只使用一条car序列，不能称为论文完整多序列复现。

---

## 二、必须使用的官方链路

严格执行：

```text
原始M3ED car数据
→ 官方 create_exp.py
→ 官方6通道IIR与exp_times
→ 官方 calc_gt.py
→ 官方 merge.py
→ 官方EV-Slim训练
→ 官方指标与可视化
```

禁止复用当前fixed10或Count10k标签。

---

## 三、输入生成

使用官方：

```text
EV-TTC-main/TTCEF/create_exp.py
```

保持：

```text
6通道signed IIR
alpha=[0.12,0.06,0.03,0.015,0.0095,0.0045]
time bin=0.2 ms
IIR输出间隔=7 ms
正事件=+1
负事件=-1
官方去畸变
原始x=[280,999]中央裁剪
坐标除以2
最终尺寸=360×360
```

必须直接使用官方生成的：

```text
exp_filts
exp_times
```

不能使用：

```text
Count10k的t_end
最近exp_times匹配
360输入下采样
自行重写IIR滤波
```

---

## 四、GT生成

使用官方：

```text
EV-TTC-main/TTCEF/calc_gt.py
```

要求：

```text
IIR输入和GT使用完全相同的exp_times
```

官方预测目标：

\[
TTC=\frac{Z}{T_z}
\]

单位：

```text
秒
```

同时生成：

```text
depth
TTC
optical flow
gt_mask
速度
角速度
```

禁止改成：

\[
\frac{T_z}{Z}
\]

也禁止复用：

```text
fixed10 signed inverse TTC
N10000_S1_360.h5标签
dense_valid_mask
event_active_mask
```

---

## 五、官方Merge与Mask

使用官方：

```text
merge.py
```

训练Mask严格按官方定义：

```text
gt_mask
& abs(exp_last_channel)>1e-3
& finite(TTC)
& TTC<100
```

同时保留官方的：

```text
速度阈值筛选
角速度阈值筛选
异常样本过滤
```

不得自行替换为当前SNN训练Mask。

---

## 六、网络结构

使用官方EV-Slim：

```text
输入：6通道
输出：1通道TTC
Encoder：[16,16,16]
ASPP：[32,16]
Decoder：[8,8,1]
```

要求：

```text
不修改第一层输入通道
不修改ASPP
不修改Decoder
不修改输出激活
不增加额外Loss
不加载当前SNN权重
```

如果官方默认从头训练，则保持：

```text
ckpt_path=null
```

---

## 七、GPU与Batch Size

本次优先使用：

```text
单张RTX 4090
batch_size=128
```

要求：

```text
不使用双卡DDP
不使用DataParallel
不把batch_size=128解释为每张GPU 128
```

目标是保持：

\[
\text{global batch size}=128
\]

启动前记录：

```text
实际使用GPU编号
显存占用
每step耗时
每epoch耗时
是否启用AMP
```

如果单张4090出现OOM：

1. 先记录完整报错和峰值显存；
2. 不静默降低batch size；
3. 优先尝试官方precision=16-mixed；
4. 若仍OOM，再使用更小micro batch配合梯度累积，使等效global batch仍为128；
5. 在报告中明确实际micro batch和accumulate_grad_batches。

---

## 八、官方20-Epoch基线

严格读取并复用：

```text
EV-TTC-main/model/conf/config.yaml
EV-TTC-main/model/conf/models/evslim_ttc.yaml
```

官方基线固定为：

```yaml
optimizer: AdamW
learning_rate: 2.0e-3
scheduler: OneCycleLR
pct_start: 0.1
epochs: 20
batch_size: 128
precision: 16-mixed
loss: masked Charbonnier
charbonnier_alpha: 0.45
smoothness_weight: 0
```

数据增强按官方启用：

```text
水平翻转
垂直翻转
RandomRotation(0,180)
flip_prob=0.3
```

这一组命名：

```text
EVTTC_Official_Car_20ep
```

它是本次单序列官方链路基线。

---

## 九、是否增加Epoch

不能直接把官方20 epochs改成更大值后称为官方复现。

先完整跑完：

```text
20 epochs
```

然后根据Validation曲线判断是否需要扩展。

### 允许扩展到40 Epoch的条件

满足以下任一情况：

```text
第20轮Validation Loss仍明显下降
最近5个Epoch的Validation Loss总体下降
Best Epoch出现在18-20附近
训练与验证均未出现明显反弹
```

### 不增加Epoch的条件

```text
Validation Loss已趋于平稳
Best Epoch明显早于第20轮
Train Loss下降但Validation Loss持续上升
Validation可视化开始恶化
```

---

## 十、40-Epoch扩展实验

若满足扩展条件，额外从头训练：

```text
40 epochs
```

命名：

```text
EVTTC_Official_Car_40ep_Extended
```

要求：

```text
重新随机初始化
重新建立OneCycleLR
max_epochs=40
scheduler总步数按40 epochs重新计算
不能直接从20-epoch checkpoint机械续训
```

除Epoch外，其余配置与20-Epoch基线保持一致：

```text
单张4090
batch_size=128
AdamW
lr=2e-3
OneCycleLR
pct_start=0.1
16-mixed
官方增强
官方Loss
```

最终报告必须同时保留：

```text
20-Epoch官方基线
40-Epoch扩展实验
```

不能只保留40-Epoch结果。

---

## 十一、单序列Train/Validation划分

由于当前只使用一条序列，采用：

```text
Train：前80%
Buffer：中间5%
Validation：后15%
```

要求：

```text
按exp_times物理时间顺序划分
Buffer不参与训练和验证
Validation不参与模型更新
```

必须标记：

```text
single-sequence debug split
```

不得称为论文官方Train/Test划分。

---

## 十二、Checkpoint与Best模型

20-Epoch组保存：

```text
latest_20ep.ckpt
best_val_loss_20ep.ckpt
```

40-Epoch组保存：

```text
latest_40ep.ckpt
best_val_loss_40ep.ckpt
```

Best模型按Validation Loss选择。

Checkpoint包含：

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
global_step
官方配置
数据划分
随机种子
GPU编号
batch size
```

---

## 十三、必须记录的指标

每个Epoch记录：

```text
train loss
validation loss
MAE
MRE
median absolute error
有效像素数
预测mean/std/min/max
NaN/Inf
learning rate
每epoch耗时
峰值显存
```

另记录：

```text
TTC<1s区域MAE
TTC<2s区域MAE
TTC<5s区域MAE
```

---

## 十四、可视化

固定选择：

```text
Train样本2个
Validation样本3个
```

保存：

```text
6个IIR通道
GT TTC
Prediction TTC
官方训练Mask
绝对误差图
预测与GT直方图
```

20-Epoch组至少保存：

```text
Epoch 0
Epoch 5
Epoch 10
Epoch 15
Epoch 20
Best Epoch
```

40-Epoch组至少保存：

```text
Epoch 0
Epoch 10
Epoch 20
Epoch 30
Epoch 40
Best Epoch
```

统一使用：

```text
相同色图
相同vmin/vmax
显示真实min/max
```

重点观察：

```text
道路纵向TTC梯度
车辆和树干边界
近场高风险区域
预测是否过度平滑
20到40 Epoch是否真正改善细节
```

---

## 十五、20与40 Epoch对照

必须生成：

```text
official_20ep_vs_40ep.csv
```

至少比较：

| 指标 | 20 Epoch | 40 Epoch |
|---|---:|---:|
| Best Epoch |  |  |
| Best Val Loss |  |  |
| Val MAE |  |  |
| Val MRE |  |  |
| TTC<1s MAE |  |  |
| TTC<2s MAE |  |  |
| 预测std |  |  |
| 边缘清晰度代理指标 |  |  |
| 总训练时间 |  |  |

最终明确判断：

```text
增加Epoch是否改善泛化
是否只改善训练集
是否出现过拟合
是否值得在后续多序列实验中采用40 Epoch
```

---

## 十六、与当前EVSlim_IIR360的对照

生成逐项对照表：

| 项目 | 当前EVSlim_IIR360 | 本次官方链路 |
|---|---|---|
| IIR输入 | 官方IIR | 官方IIR |
| 输入时间戳 | 最近匹配 | 与GT完全同exp_times |
| 标签 | signed inverse TTC | 官方TTC秒值 |
| Mask | 当前统一Mask | 官方Mask |
| Batch size | 32 | 单GPU 128 |
| 数据增强 | 关闭 | 官方开启 |
| Epoch | 20 | 官方20，必要时扩展40 |
| 数据划分 | 500-Block | 单序列官方链路debug划分 |
| 网络 | EV-Slim | 官方EV-Slim |

---

## 十七、通过标准

本任务通过需满足：

```text
官方create_exp.py成功运行
官方calc_gt.py成功运行
IIR与GT时间戳完全一致
官方merge.py成功生成训练数据
单张4090、global batch 128完成训练
官方20-Epoch基线完成
训练与验证无NaN/Inf
验证预测明显优于随机初始化
可视化能恢复主要TTC结构
```

40-Epoch组不是强制通过条件，只在满足扩展条件时运行。

---

## 十八、必须生成的报告

```text
car单序列官方EV-TTC数据生成报告.md
car单序列官方EV-TTC训练配置核对.md
car单序列官方EV-TTC_20Epoch训练结果.md
car单序列官方EV-TTC_40Epoch扩展结果.md
car单序列官方EV-TTC_20与40Epoch对比.md
car单序列官方EV-TTC可视化分析.md
car单序列官方EV-TTC复现通过或失败判定.md
```

如果不满足40-Epoch扩展条件，仍生成：

```text
car单序列官方EV-TTC_40Epoch扩展必要性判定.md
```

说明为什么没有继续。

---

## 十九、最终必须回答

1. IIR是否由官方create_exp.py生成；
2. IIR与GT是否使用完全相同的exp_times；
3. 预测目标是否为官方TTC秒值；
4. Mask是否严格采用官方定义；
5. 网络是否未修改；
6. 是否使用单张RTX 4090；
7. global batch size是否严格为128；
8. 20-Epoch官方训练是否完成；
9. 第20轮时Validation是否仍在下降；
10. 是否有必要运行40-Epoch扩展；
11. 若运行40 Epoch，是否从头训练并重建OneCycleLR；
12. 40 Epoch是否真实改善Validation和局部结构；
13. 当前结果是否优于之前signed inverse TTC适配版本；
14. 当前结果能否称为单序列官方链路复现；
15. 为什么仍不能称为论文完整复现。

完成后停止，不自动进入多序列正式复现。
