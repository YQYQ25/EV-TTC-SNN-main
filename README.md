# EV-TTC-SNN

本仓库记录基于 M3ED 的事件相机 TTC 数据构建、EV-TTC 官方 EV-Slim 对照实验，以及将 MAVLab event-flow 的 LIF-EV-FlowNet 迁移为 signed inverse TTC 回归网络的代码。

## 目录

- `EV-TTC-SNN-main/snn_ttc/`：本项目的 SNN-TTC 数据处理、模型、训练、审计与可视化代码。
- `EV-TTC-SNN-main/guidance_files/`：实验设计与实现说明。
- `EV-TTC-main/model/`：EV-TTC EV-Slim 网络及相关训练模块，包含本项目使用的可配置输入通道实现。
- `EV-TTC-main/TTCEF/`：官方 IIR 表示与 TTC 标签生成参考实现。
- `EV-TTC-main/tools/`：本项目用于 EV-TTC 对照、评估和数据下载的辅助脚本。
- `event_flow-main/`：LIF-EV-FlowNet 所依赖的 MAVLab event-flow 源码、配置与训练工具。

## 不包含的内容

M3ED、MVSEC、ECD 等原始数据，生成的 H5，checkpoint，ONNX/engine，训练日志，图像可视化和虚拟环境均不包含在 Git 仓库中。脚本中的本地数据路径需要根据使用者环境调整。

## 主要依赖

Python、PyTorch、NumPy、SciPy、OpenCV、h5py、hdf5plugin、Matplotlib、Numba、tqdm，以及在重投影场景需要的 `projectaria-tools`。

## 第三方来源与许可证

- `EV-TTC-main/` 保留其 MIT 许可证，版权归原作者 Anthony Bisulco。
- `event_flow-main/` 保留其 MIT 许可证，版权归 TU Delft；其中部分实现注明改编自 UZH-RPG E2VID。

本仓库不发布数据集和预训练权重。使用 M3ED 数据时，请遵循原始数据集的许可和使用条款。
