# guidance_files 指导文件索引

整理时间：2026-07-19

本目录用于保存阶段性任务指导文件。文件已按实验主题归类，方便后续继续做 SNN-TTC 迁移、M3ED 标签生成和空间方案对比。

## 目录结构

| 目录 | 内容 | 阅读优先级 |
|---|---|---|
| `00_overview/` | Event Flow SNN 迁移到 EV-TTC 的总体思路和阶段总结 | 先读 |
| `01_m3ed_1k_roi_debug/` | M3ED 固定事件数、ROI、TTC 标签、mask、监督密度等基础数据生成调试 | 做数据生成时读 |
| `02_skatepark_audit/` | Skatepark 序列的多事件数标签验证、官方筛选条件和负 TTC 统计 | 复查 Skatepark 时读 |
| `03_car_spatial_ablation/` | Car 数据集三种空间方案、EVTTC 官方对齐、既有文件复用和最小修正 | 做 car 空间方案时读 |

## 00_overview

| 文件 | 含义 |
|---|---|
| `event_flow_transfer_EVTTC总体思路.md` | 将 event_flow 中 SNN 模型迁移到 EV-TTC 输出 TTC map 的总体路线 |
| `总体思路的报告文件.md` | 总体思路对应的阶段性报告或补充说明 |

## 01_m3ed_1k_roi_debug

| 文件 | 含义 |
|---|---|
| `M3ED_1k事件分片与TTC标签调试集实现指令.md` | 第一版 1k 事件分片与 TTC 标签调试集实现 |
| `M3ED固定ROI内1k事件分片与TTC标签生成修正指令.md` | 固定 ROI 后，对 TTC 标签和 mask 生成方式的修正 |
| `M3ED固定1k事件时间尺度与ROI切换验证指令.md` | 检查固定 1k 事件的时间跨度，以及 ROI 切换策略 |
| `SuperviseValid与空标签Step原因及Block监督密度审计指令.md` | 分析 mask 全黑、空标签 step、监督密度不足等问题 |

## 02_skatepark_audit

| 文件 | 含义 |
|---|---|
| `Skatepark多事件数TTC标签与超时换ROI整夜验证指令_修正版.md` | Skatepark 上 N=5000/10000/15000/20000 等多事件数完整验证 |
| `EVTTC官方Skatepark筛选与负TTC统计指令.md` | 按 EV-TTC 官方条件筛选 Skatepark，并统计正负 TTC |

## 03_car_spatial_ablation

| 文件 | 含义 |
|---|---|
| `Car数据集MAVLab固定事件数与EVTTC官方方案对比验证指令.md` | Car 数据集上固定事件数方案与官方 EV-TTC 方案的初步对比 |
| `Car多事件数三种空间方案与EVTTC官方对比指令.md` | S1/S2/S3 三种空间方案与官方样本进行多事件数对比 |
| `Car三种空间方案既有文件复用与修改审计指令.md` | 审计已有文件哪些能复用，哪些需要修正或作废 |
| `Car三种空间方案既有实验最小修正执行指令.md` | 在已有结果基础上做最小修正，得到当前正式 fixed-N 主线 |

## 当前建议

后续如果继续做 SNN-TTC 数据生成，优先看 `00_overview/` 和 `01_m3ed_1k_roi_debug/`。

如果继续做 car 空间方案和 official 对齐，优先看 `03_car_spatial_ablation/`，并结合：

```text
../debug_sets/[5]car_spatial_ablation/00_目录说明与清理建议.md
```

如果只是复查历史过程，`02_skatepark_audit/` 和旧报告可以作为参考，不建议直接作为当前主实验结论。
