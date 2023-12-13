# 项目介绍
项目使用，请移步[知乎](https://zhuanlan.zhihu.com/p/656537283)

## 测试ONNX模型的精度

防止发生ONNX转换时发生严重的精度损失

### 测试MaxQ的模型

```shell
python evaluate.py ./dataset/imagenet --onnx-path model/MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx --save_dir log/MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx
```

```text 
[2023-10-04 21:58:48] Test: [0/782]	Time 2.697 (2.697)	Loss 1.3634 (1.3634)	Prec@1 93.750 (93.750)	Prec@5 96.875 (96.875)
[2023-10-04 21:58:57] Test: [100/782]	Time 0.074 (0.113)	Loss 1.4562 (1.6241)	Prec@1 89.062 (83.184)	Prec@5 96.875 (95.606)
[2023-10-04 21:59:05] Test: [200/782]	Time 0.102 (0.099)	Loss 1.6271 (1.6149)	Prec@1 85.938 (83.209)	Prec@5 96.875 (96.160)
[2023-10-04 21:59:14] Test: [300/782]	Time 0.041 (0.096)	Loss 1.5237 (1.6168)	Prec@1 84.375 (82.916)	Prec@5 95.312 (96.268)
[2023-10-04 21:59:22] Test: [400/782]	Time 0.051 (0.093)	Loss 1.5397 (1.7102)	Prec@1 82.812 (80.510)	Prec@5 98.438 (95.254)
[2023-10-04 21:59:31] Test: [500/782]	Time 0.058 (0.092)	Loss 1.3616 (1.7604)	Prec@1 90.625 (79.351)	Prec@5 98.438 (94.617)
[2023-10-04 21:59:39] Test: [600/782]	Time 0.103 (0.090)	Loss 1.6276 (1.8015)	Prec@1 87.500 (78.450)	Prec@5 98.438 (94.171)
[2023-10-04 21:59:48] Test: [700/782]	Time 0.073 (0.089)	Loss 1.8265 (1.8354)	Prec@1 78.125 (77.619)	Prec@5 95.312 (93.710)
[2023-10-04 22:00:00]  * Prec@1 77.480 Prec@5 93.682 Error@1 22.520
[2023-10-04 22:00:00] => Epoch: 119, LR: 0.0000, Acc: 77.48%, Best Acc: 77.58%
```

```text
# 原始精度77.58%, 说明Torch转ONNX没有明显精度损失
[2023-10-12 15:07:48] Test: [49300/50000]	Prec@1 77.493%	Prec@5 93.665%
[2023-10-12 15:07:54] Test: [49400/50000]	Prec@1 77.490%	Prec@5 93.674%
[2023-10-12 15:08:01] Test: [49500/50000]	Prec@1 77.529%	Prec@5 93.687%
[2023-10-12 15:08:08] Test: [49600/50000]	Prec@1 77.567%	Prec@5 93.698%
[2023-10-12 15:08:16] Test: [49700/50000]	Prec@1 77.590%	Prec@5 93.706%
[2023-10-12 15:08:23] Test: [49800/50000]	Prec@1 77.624%	Prec@5 93.715%
[2023-10-12 15:08:29] Test: [49900/50000]	Prec@1 77.633%	Prec@5 93.717%
[2023-10-12 15:08:36] Test: [50000/50000]	Prec@1 77.576%	Prec@5 93.708%
```

### 测试SR-STE的模型

```shell
python evaluate.py ./dataset/imagenet --onnx-path model/SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx --save_dir log/SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx
```

```text
Test: [0/391]	Time 7.514 (7.514)	Loss 0.4347 (0.4347)	Prec@1 90.625 (90.625)	Prec@5 96.875 (96.875)
Test: [100/391]	Time 0.249 (0.538)	Loss 0.5370 (0.6627)	Prec@1 87.500 (82.944)	Prec@5 96.875 (96.078)
Test: [200/391]	Time 0.694 (0.491)	Loss 0.6714 (0.7845)	Prec@1 82.812 (80.309)	Prec@5 95.312 (95.021)
Test: [300/391]	Time 0.248 (0.470)	Loss 0.8241 (0.9012)	Prec@1 82.812 (77.928)	Prec@5 93.750 (93.724)
 * Prec@1 76.896 Prec@5 93.314
```

```text 
# 原始精度76.896%, 说明Torch转ONNX没有明显精度损失
[2023-10-12 18:04:03] Test: [49300/50000]	Prec@1 76.822%	Prec@5 93.268%
[2023-10-12 18:04:10] Test: [49400/50000]	Prec@1 76.814%	Prec@5 93.275%
[2023-10-12 18:04:16] Test: [49500/50000]	Prec@1 76.857%	Prec@5 93.289%
[2023-10-12 18:04:22] Test: [49600/50000]	Prec@1 76.893%	Prec@5 93.296%
[2023-10-12 18:04:28] Test: [49700/50000]	Prec@1 76.918%	Prec@5 93.304%
[2023-10-12 18:04:35] Test: [49800/50000]	Prec@1 76.950%	Prec@5 93.311%
[2023-10-12 18:04:41] Test: [49900/50000]	Prec@1 76.960%	Prec@5 93.319%
[2023-10-12 18:04:47] Test: [50000/50000]	Prec@1 76.896%	Prec@5 93.314%
```

## 测试ONNX模型的精度INT8

### 测试MaxQ的模型

```shell
python evaluate.py ./dataset/imagenet \
  --onnx-path ../MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim/outputs/calibration/resnet50_fake_quant_model.onnx \
  --save_dir ./log/MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx-int8
```

```text
# 原始精度77.58%
[2023-10-12 23:17:16] Test: [49600/50000]       Prec@1 77.127%  Prec@5 93.546%
[2023-10-12 23:17:38] Test: [49700/50000]       Prec@1 77.151%  Prec@5 93.553%
[2023-10-12 23:18:03] Test: [49800/50000]       Prec@1 77.187%  Prec@5 93.562%
[2023-10-12 23:18:30] Test: [49900/50000]       Prec@1 77.196%  Prec@5 93.565%
[2023-10-12 23:18:52] Test: [50000/50000]       Prec@1 77.142%  Prec@5 93.554%
[2023-10-12 23:18:53] [INFO] ONNX PATH: ../MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim/outputs/calibration/resnet50_fake_quant_model.onnx
[2023-10-12 23:18:53] [INFO] ResNet18 before quantize top1:77.14% top5:93.55%
```

### 测试SR-STE的模型

```shell
python evaluate.py ./dataset/imagenet \
  --onnx-path ../SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim/outputs/calibration/resnet50_fake_quant_model.onnx \
  --save_dir ./log/SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx-int8
```

```text
# 原始精度76.896%
[2023-10-13 02:27:36] Test: [49600/50000]       Prec@1 76.611%  Prec@5 93.206%
[2023-10-13 02:27:59] Test: [49700/50000]       Prec@1 76.636%  Prec@5 93.213%
[2023-10-13 02:28:22] Test: [49800/50000]       Prec@1 76.669%  Prec@5 93.221%
[2023-10-13 02:28:46] Test: [49900/50000]       Prec@1 76.679%  Prec@5 93.230%
[2023-10-13 02:29:09] Test: [50000/50000]       Prec@1 76.610%  Prec@5 93.222%
[2023-10-13 02:29:09] [INFO] ONNX PATH: ../SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim/outputs/calibration/resnet50_fake_quant_model.onnx
[2023-10-13 02:29:09] [INFO] ResNet18 before quantize top1:76.61% top5:93.22%
```

