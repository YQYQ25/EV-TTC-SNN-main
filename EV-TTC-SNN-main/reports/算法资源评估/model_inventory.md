# 模型清单

本评估只读取既有 checkpoint 和固定测试样本。

## 官方 ANN EV-Slim

- 模型入口：`/home/hello/research_project/event+SNN+TTC/EV-TTC-main/model/evslim.py`，`evslim.EVSlim`
- checkpoint：`/home/hello/research_project/event+SNN+TTC/EV-TTC-SNN-main/debug_sets/[13]official_car_single_sequence_evttc_reproduction/EVTTC_Official_Car_20ep/best_val_loss_20ep.ckpt`
- checkpoint 精度：FP32 权重；运行时分别测 FP32 与 CUDA AMP FP16。
- 输入：`[1, 6, 360, 360]`，H5 exp_filts直接输入
- 输出：单通道稠密 TTC / signed inverse TTC map，形状由实际前向统计。
- 时间步：`1`；状态管理：无状态
- 自定义/状态算子：无；使用 Conv、BatchNorm、ReLU、ASPP 和普通上采样/卷积层。
- 推理入口：本资源评估脚本；数据格式与原训练/验证入口一致。

## Hybrid SNN EV-Slim

- 模型入口：`/home/hello/research_project/event+SNN+TTC/EV-TTC-main/model/hybrid_snn_evslim.py`，`hybrid_snn_evslim.HybridSNNEVSlim`
- checkpoint：`/home/hello/research_project/event+SNN+TTC/EV-TTC-SNN-main/reports/06_真实数据端到端验证/checkpoints/best.pt`
- checkpoint 精度：FP32 权重；运行时分别测 FP32 与 CUDA AMP FP16。
- 输入：`[1, 3, 6, 360, 360]`，H5 exp_filts直接输入
- 输出：单通道稠密 TTC / signed inverse TTC map，形状由实际前向统计。
- 时间步：`3`；状态管理：每个3-step窗口前reset_states
- 自定义/状态算子：实际为两层 LIF（lif1、lif2）加第三层实值累加器；ASPP 与 decoder 仅在序列末执行一次。
- 推理入口：本资源评估脚本；数据格式与原训练/验证入口一致。

## MAVLab LIF-EV-FlowNet-TTC

- 模型入口：`/home/hello/research_project/event+SNN+TTC/EV-TTC-SNN-main/snn_ttc/models/lif_evflownet_ttc.py`，`snn_ttc.models.lif_evflownet_ttc.LIFEVFlowNetTTC`
- checkpoint：`/data/evttc_storage/event+SNN+TTC/EV-TTC-SNN-main/debug_sets_offloaded/[10]lif_evflownet_ttc_s2_n10k_500block_pipeline/checkpoints/best_val_mae.pt`
- checkpoint 精度：FP32 权重；运行时分别测 FP32 与 CUDA AMP FP16。
- 输入：`[1, 10, 2, 128, 128]`，event_cnt * 0.3
- 输出：单通道稠密 TTC / signed inverse TTC map，形状由实际前向统计。
- 时间步：`10`；状态管理：每个10-step block前reset_states，block末detach但本测量无反传
- 自定义/状态算子：ConvLIF、ConvLIFRecurrent、脉冲残差块和脉冲上采样块；完整10步都会更新状态。
- 推理入口：本资源评估脚本；数据格式与原训练/验证入口一致。
