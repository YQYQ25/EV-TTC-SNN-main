我们准备开发一套基于 M3ED 的 LIF-EV-FlowNet 稠密 TTC 训练流程。

总体目标：
1. 从 M3ED 原始事件流中，按连续 1k 事件划分时间步；
2. 每 10 个时间步组成一次 BPTT；
3. 使用 EV-TTC 的几何方法生成 TTC、inverse TTC、depth 和 valid mask；
4. 支持分片起点和结束时刻两套标签；
5. event、TTC、depth、mask 使用完全一致的裁剪和翻转；
6. 对不满足速度、角速度或标签有效性条件的时间步：
   - 事件仍正常前向传播并更新 SNN 状态；
   - 该时间步不参与监督损失；
7. 第 10 步结束后，对有效时间步和有效像素上的损失统一归一化，再执行一次 BPTT；
8. 训练和推理都保持连续 1k 事件一步；
9. 先完成小规模数据生成、可视化检查和 ANN/SNN 过拟合，不立即全量训练。

当前阶段不要修改代码。请先检查工作区中的现有项目，并输出一份：
code_audit_for_m3ed_ttc.md

请重点查找并分析以下代码。

一、MAVLab event_flow 相关
- configs/train_SNN.yml
- dataloader/h5.py
- dataloader/base.py
- dataloader/encodings.py
- train_flow.py
- loss/flow.py
- models/model.py
- models/spiking_submodules.py
- 其他与 reset_states、detach_states、事件分片、数据增强、BPTT、梯度裁剪有关的文件

二、EV-TTC 相关
- TTCEF/create_exp.py
- TTCEF/calc_gt.py
- TTCEF/merge.py
- model/data/ttc_dm.py
- 其他与标定、去畸变、深度重投影、TTC生成、mask、速度/角速度筛选有关的文件

三、当前工作区已经存在的自开发代码
请搜索所有与以下关键词相关的文件：
- M3ED
- TTC
- inverse_ttc
- event_cnt
- 1k events
- BPTT
- reset_states
- depth
- mask
- crop
- undistort
- pose interpolation

报告必须包含：

1. 项目目录树
只展开相关文件，不要列无关目录。

2. 每个关键文件的职责
说明其输入、输出、关键函数和调用关系。

3. 当前实际数据流
分别画出：
- MAVLab训练流程；
- EV-TTC数据生成流程；
- 当前自开发流程。

4. 关键张量或数组形状
例如：
- event_cnt
- event_list
- depth
- TTC
- mask
- 网络输入
- 多尺度输出

5. 时间组织方式
明确说明：
- 每次读取多少事件；
- 何时累计到10步；
- 何时 backward；
- 何时 detach_states；
- 何时 reset_states；
- 序列末尾不足10步如何处理。

6. 空间处理
明确说明：
- 去畸变发生在哪里；
- 裁剪和下采样发生在哪里；
- 翻转参数如何生成；
- 同一连续序列内增强参数是否保持一致；
- 标签和事件是否严格同步。

7. TTC标签生成
明确说明：
- 标签对应 start_time 还是 end_time；
- 深度重投影到哪个时刻；
- T、Omega表达在哪个坐标系；
- Tz符号；
- 负TTC、低速和最大TTC如何处理；
- mask最终由哪些条件构成。

8. 当前代码中可直接复用和必须重写的部分
分成：
- 可原样复用；
- 小幅修改；
- 必须新写；
- 当前存在风险或不确定的部分。

9. 提供关键源码
不要只给总结。对每个关键结论，附对应文件路径、函数名和必要源码片段。

10. 不要现在开始实现
先完成代码审计，等待下一步指令。