# Hybrid SNN-EV-Slim 随机张量单元测试

- 总体结果：通过
- 设备：`cuda:0`
- 输入：`[2, 3, 6, 64, 64]`
- 输出：`[2, 1, 64, 64]`
- LIF1/LIF2发放率：`{'lif1': 0.20537312825520834, 'lif2': 0.20592498779296875}`
- accumulator：`{'initialized': True, 'mean': 5.180481821298599e-09, 'std': 1.75264310836792, 'min': -7.502952575683594, 'max': 9.042036056518555, 'finite': True}`
- prediction：`{'mean': 0.09698770195245743, 'std': 0.33104458451271057, 'min': -1.632904291152954, 'max': 1.6201635599136353}`
- 梯度范数：`{'conv1': 0.09932850526764204, 'conv2': 0.18212309718136183, 'conv3': 0.19580150423368314, 'aspp': 0.2713725624365668, 'ttc_head': 0.28419356213943026, 'snn_encoder': 0.2853441701967643, 'ann_backend': 0.4217125616862688}`
- 时间步输入梯度：`[{'gradient_sum': 0.8078593015670776, 'gradient_mean': 1.6435940779047087e-05, 'gradient_max': 0.00010504025703994557, 'finite': True}, {'gradient_sum': 0.8149858117103577, 'gradient_mean': 1.658092878642492e-05, 'gradient_max': 0.00010045324597740546, 'finite': True}, {'gradient_sum': 0.8011199235916138, 'gradient_mean': 1.6298827176797204e-05, 'gradient_max': 0.00010184646816924214, 'finite': True}]`
- 时间步置零输出 L1 差异：`{'zero_t0': 0.1368524730205536, 'zero_t1': 0.17563146352767944, 'zero_t2': 0.15405988693237305}`
- AMP：`{'status': 'passed', 'loss': 0.8341712951660156, 'parameter_changed_count': 57, 'gradients_finite': True, 'scaler_scale': 65536.0, 'checkpoint_scaler_restored': True}`
- 更新参数数量：`57`

详细检查见同目录 `hybrid_snn_evslim_unit_test.json`。
