import os

methods = ["SR-STE", "MaxQ"]
scripts_file = 'quantization_scripts.sh'
os.system(f'rm {scripts_file}')
N = 2
M = 4

batch_sizes = [1, 2, 4, 8, 16]
for method in methods:
    for batch_size in batch_sizes:
        onnx_path = os.path.join('..', method, f'{N}-{M}', f"batch_size-{batch_size}-N-{N}-M-{M}-resnet50-sim.onnx")
        os.system(f"echo python quantization.py ./dataset/imagenet --onnx-path {onnx_path} --batch-size  {batch_size} >> {scripts_file}")
