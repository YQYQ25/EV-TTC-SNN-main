# 基于脉冲神经网络的事件光流自监督学习

本工作已被 NeurIPS 2021 接收：[[论文](https://proceedings.neurips.cc/paper/2021/hash/39d4b545fb02556829aab1db805021c3-Abstract.html)，[视频](https://www.youtube.com/watch?v=T7-9GGYnuZ4&ab_channel=MAVLabTUDelft)]。

如在学术研究中使用本代码，请引用我们的工作：

```bibtex
@article{hagenaarsparedesvalles2021ssl,
  title={Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks},
  author={Hagenaars, Jesse and Paredes-Vall\'es, Federico and de Croon, Guido},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

本代码可用于复现论文第 4.1 节实验及其结果。

<!-- &nbsp; -->
<img src=".readme/flow.gif" width="880" height="220" />
<!-- &nbsp; -->

#

## 使用方法

本项目要求 Python >= 3.7.3，并强烈建议使用虚拟环境。如果尚未安装环境管理工具，推荐使用 `pyenv`。可通过以下命令安装：

```bash
curl https://pyenv.run | bash
```

确保你的 `~/.bashrc` 文件包含以下内容：

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

随后重启终端，并运行：

```bash
pyenv update
```

使用 `pyenv` 配置环境时，首先安装所需的 Python 版本，并确保安装过程成功，即未出现错误或警告：

```bash
pyenv install -v 3.7.3
```

安装完成后，创建虚拟环境并安装所需依赖：

```bash
pyenv virtualenv 3.7.3 event_flow
pyenv activate event_flow

pip install --upgrade pip==20.0.2

cd event_flow/
pip install -r requirements.txt
```

### 下载数据集

本工作使用了多个数据集：

- `event_flow/datasets/data/training`：[UZH-FPV 无人机竞速数据集](https://fpv.ifi.uzh.ch/)（Delmerico，ICRA 2019）
- `event_flow/datasets/data/MVSEC`：[多视角立体事件相机数据集](https://daniilidis-group.github.io/mvsec/)（Zhu，RA-L 2018）
- `event_flow/datasets/data/ECD`：[事件相机数据集](http://rpg.ifi.uzh.ch/davis_data.html)（Mueggler，IJRR 2017）
- `event_flow/datasets/data/HQF`：[高质量帧数据集](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720528.pdf)（Stoffregen 和 Scheerlinck，ECCV 2020）

可从[此处](https://1drv.ms/u/s!Ah0kx0CRKrAZjx-EEIzfo8iqBDro?e=TIoxG9)下载已转换为项目所需 HDF5 格式的数据集。下载后应将其放置在 `event_flow/datasets/data/` 目录下，具体目录结构如上所示。

下载文件大小：19.4 GB。解压后大小：94 GB。

有关这些文件结构的详细说明，请参阅 `event_flow/datasets/tools/`。

### 下载模型

可从[此处](https://1drv.ms/u/s!Ah0kx0CRKrAZjyD2MUxoRQQ-O0TI?e=MUlhCx)下载预训练模型，并将其放置在 `event_flow/mlruns/` 目录下。

本项目使用 [MLflow](https://www.mlflow.org/docs/latest/index.html#) 跟踪实验。若要查看已有模型、相关实验信息及评估指标，请在项目根目录下运行：

```bash
mlflow ui
```

然后使用浏览器访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)。

## 推理

若要使用 MVSEC 数据集中的事件序列估计光流，并计算平均端点误差和异常值比例，请运行：

```bash
python eval_flow.py <model_name> --config configs/eval_MVSEC.yml

# 示例：
python eval_flow.py LIFFireNet --config configs/eval_MVSEC.yml
```

其中，`<model_name>` 表示待评估的 MLflow 运行名称。若某次运行没有名称，例如自行训练的模型，也可以使用其运行 ID 进行评估。运行 ID 同样可在 MLflow 中查看。

若要使用 ECD 或 HQF 数据集中的事件序列估计光流，请运行：

```bash
python eval_flow.py <model_name> --config configs/eval_ECD.yml
python eval_flow.py <model_name> --config configs/eval_HQF.yml

# 示例：
python eval_flow.py LIFFireNet --config configs/eval_ECD.yml
```

ECD 和 HQF 数据集不包含光流真值。因此，本项目使用自监督指标 [FWL](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720528.pdf)（Stoffregen 和 Scheerlinck，ECCV 2020）以及作者提出的 RSAT 指标（见附录 C），评估事件光流估计结果的质量。

评估结果会以 MLflow artifact 的形式保存。

在 `configs/` 目录中，可以找到与上述脚本对应的配置文件，并修改推理设置，例如输入事件数量以及是否启用可视化。

## 训练

运行：

```bash
python train_flow.py --config configs/train_ANN.yml
python train_flow.py --config configs/train_SNN.yml
```

上述两条命令分别用于训练传统人工神经网络 ANN（默认模型为 FireNet）和脉冲神经网络 SNN（默认模型为 LIF-FireNet）。

在 `configs/` 目录中，可以找到对应的配置文件，并修改训练设置，例如模型类型、输入事件数量以及是否启用可视化。其他可用模型请参阅 `models/model.py`。

**本项目实验使用的 batch size 为 8。请根据实际计算资源适当减小该数值。**

训练期间及训练完成后，均可通过 MLflow 查看相应运行信息。

## 卸载 pyenv

使用完本项目代码后，可通过以下步骤卸载 `pyenv`：

1. 删除 `~/.bashrc` 文件中与 `pyenv` 相关的配置行。
2. 删除 `pyenv` 的根目录。该操作会同时删除安装在 `$HOME/.pyenv/versions/` 目录下的所有 Python 版本：

```bash
rm -rf $HOME/.pyenv/
```
