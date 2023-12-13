import os

methods = ["SR-STE", "MaxQ"]
N = 2
M = 4
framework = 5
soc_version = 'Ascend310B4'
batch_sizes = [1, 2, 4, 8, 16]

scripts_name = 'atc_fp16.sh'
os.system(f'rm {scripts_name}')

for method in methods:
    for batch_size in batch_sizes:
        onnx_path = os.path.join('..', method, f'{N}-{M}', f"batch_size-{batch_size}-N-{N}-M-{M}-resnet50-sim.onnx")
        output = os.path.dirname(onnx_path)
        output = os.path.join(output, os.path.basename(onnx_path).strip(".onnx"))
        os.system(f"echo mkdir {output} >> {scripts_name}")
        output_file = os.path.join(output, 'resnet50_fp16')
        os.system(f"echo atc --model {onnx_path} --framework {framework} --output {output_file} "
                  f"--soc_version {soc_version} >> {scripts_name}")
