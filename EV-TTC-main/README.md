# **EV-TTC: Event-Based Time-To-Collision Estimation**
[![Paper](https://img.shields.io/badge/paper-EV_TTC-blueviolet.svg)](https://ieeexplore.ieee.org/document/10979412)

EV-TTC is a high-speed **Time-To-Collision (TTC)** estimation system created using a fast multi-temporal scale event representation and a slim dilated convolution network. This repository provides the complete pipeline for TTC estimation, including **C++ ROS2 nodes**, **neural network training**, and **dataset preparation**.


https://github.com/user-attachments/assets/4d9b3bb7-925e-4cbf-9c4e-5b32667ced9f


---

## **Key Contributions**
1. **High-Speed Multi-Temporal Scale Filter**  
   - Runs with a latency of **3.3ms** at **75 Million Events Per Second** on a **Jetson Orin NX 16GB** with events from **720x720 image resolution**.
2. **EV-Slim Neural Network**  
   - A lightweight, high-speed network optimized with TensorRT for minimal latency.
3. **$T^2CEF$ Dataset**  
   - A new dataset created from the **M3ED dataset**, consisting of ground truth optical flow and TTC for event camera sequences.

---

## **Repository Structure**

### **1. `ev_ttc/`: ROS2 Node for Real-Time TTC Estimation**
##### This folder contains the **C++ ROS2 implementation** for real-time TTC estimation. For more details, refer to the [ev_ttc README](ev_ttc/README.md).
---

### **2. `model/`: Neural Network Training and Inference**
##### This folder contains the **PyTorch Lightning pipeline** for training and evaluating the EV-Slim neural network. For more details, refer to the [model README](model/README.md).
---
### **3. `TTCEF/`: Dataset Preparation**
##### This folder contains scripts for preparing the $T^2CEF$ dataset from the M3ED dataset. For more details, refer to the [TTCEF README](TTCEF/README.md).
---

### Citation
```
@ARTICLE{ev_ttc,
  author={Bisulco, Anthony and Kumar, Vijay and Daniilidis, Kostas},
  journal={IEEE Robotics and Automation Letters}, 
  title={EV-TTC: Event-Based Time to Collision under Low Light Conditions}, 
  year={2025},
  pages={1-8},
  doi={10.1109/LRA.2025.3565150}}
```
If you use $T^2CEF$, please additionally cite M3ED:
```
@INPROCEEDINGS{m3ed,
  author={Chaney, Kenneth and Cladera, Fernando and Wang, Ziyun and Bisulco, Anthony and Hsieh, M. Ani and Korpela, Christopher and Kumar, Vijay and Taylor, Camillo J. and Daniilidis, Kostas},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog. Workshop}, 
  title={{M3ED}: Multi-Robot, Multi-Sensor, Multi-Environment Event Dataset}, 
  month={July},
  year={2023}}

```
