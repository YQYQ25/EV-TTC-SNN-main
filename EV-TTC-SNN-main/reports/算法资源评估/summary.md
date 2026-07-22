# 算法资源评估摘要

RTX 4090 数据仅用于三种算法的同平台相对比较，不能直接换算为 Orin、RK3588、FPGA 或类脑芯片功耗。

| 模型 | 参数量 | MACs | 完整SNN仿真MACs | 峰值显存 | FP16延迟 | 动态功率 | 能量/次 | 平均发放率 | SynOps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方 ANN EV-Slim | 41,673 | 5.322 GMAC | - | 66.9 MiB | 1.494 ms | 300.65 W | 477.968 mJ | nan | nan |
| Hybrid SNN EV-Slim | 41,737 | 6.740 GMAC | 6.740 GMAC | 112.5 MiB | 2.420 ms | 242.66 W | 574.001 mJ | 0.0467 | 8.367e+07 |
| MAVLab LIF-EV-FlowNet-TTC | 20,398,340 | 44.683 GMAC | 44.683 GMAC | 161.1 MiB | 23.561 ms | 111.76 W | 2641.694 mJ | 0.3096 | nan |

| 模型 | 30 Hz需求 | 60 Hz需求 | 100 Hz需求 | 建议最低RAM | 推荐硬件级别 | 主要风险 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 官方 ANN EV-Slim | 159.66 GMAC/s | 319.31 GMAC/s | 532.19 GMAC/s | 512 MiB | Jetson Orin Nano级 | ASPP空洞卷积、360分辨率和内存带宽可能限制实际FPS |
| Hybrid SNN EV-Slim | 202.21 GMAC/s | 404.41 GMAC/s | 674.02 GMAC/s | 512 MiB | Jetson Orin Nano级 | LIF状态与ANN ASPP混合；NPU算子兼容性需要验证 |
| MAVLab LIF-EV-FlowNet-TTC | 1340.49 GMAC/s | 2680.99 GMAC/s | 4468.31 GMAC/s | 512 MiB | FPGA/类脑专用映射 | PyTorch稠密仿真不具事件稀疏加速；递归LIF/自定义状态算子需部署适配 |

## 口径

- `1 MAC = 2 FLOPs`；MAC 只统计 Conv2d、ConvTranspose2d 与 Linear。
- SNN GPU MAC 是普通 PyTorch 稠密卷积仿真的实际层调用总量，不因低发放率而减少。
- SynOps 是理想事件驱动硬件代理指标；MAVLab递归网络的连接拓扑未从封装层可靠恢复时以 `NaN` 标注，不作猜测。
- INT8 未测试：当前三模型没有已验证的可靠 INT8 导出路径。
