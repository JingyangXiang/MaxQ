# N:M 模型部署(目前加速的N:M稀疏模式有限)

## 导出ONNX模型

导出ONNX的模型时, 代码绘制行
1. 导出ONNX模型
2. 简化ONNX模型
3. 检验简化后的ONNX模型和原先的torch输出是否基本保持相同

```shell
python export_onnx.py --pretrained ./deploy/MaxQ/1-4/best.resnet50.2023-10-03-7285.pth.tar --N 1 --M 4
```

```shell
INFO:[ONNXOPTIMIZER]:batch_size-1-N-1-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-2-N-1-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-4-N-1-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-8-N-1-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-16-N-1-M-4-resnet50-sim.onnx   all_close: True
```

```shell
python export_onnx.py --pretrained ./deploy/MaxQ/2-4/best.resnet50.2023-10-03-8617.pth.tar --N 2 --M 4
```

```shell
INFO:[ONNXOPTIMIZER]:batch_size-1-N-2-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-2-N-2-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-4-N-2-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-8-N-2-M-4-resnet50-sim.onnx    all_close: True
INFO:[ONNXOPTIMIZER]:batch_size-16-N-2-M-4-resnet50-sim.onnx   all_close: True
```