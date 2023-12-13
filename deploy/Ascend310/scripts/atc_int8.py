import os

methods = ["SR-STE", "MaxQ"]
N = 2
M = 4
framework = 5
soc_version = 'Ascend310B4'
batch_sizes = [1, 2, 4, 8, 16]

scripts_name = 'atc_int8.sh'
os.system(f'rm {scripts_name}')

for method in methods:
    for batch_size in batch_sizes:
        onnx_path_base = os.path.join('..', method, f'{N}-{M}', f"batch_size-{batch_size}-N-{N}-M-{M}-resnet50-sim")
        onnx_path = os.path.join(onnx_path_base, 'outputs', 'calibration', 'resnet50_deploy_model.onnx')
        output = os.path.dirname(onnx_path)

        output_file_dense = os.path.join(onnx_path_base, 'resnet50_int8_dense')
        os.system(f"echo atc --model {onnx_path} --framework {framework} --output {output_file_dense} "
                  f"--soc_version {soc_version} --sparsity 0 >> {scripts_name}")

        output_file_sparse = os.path.join(onnx_path_base, 'resnet50_int8_sparse')
        os.system(f"echo atc --model {onnx_path} --framework {framework} --output {output_file_sparse} "
                  f"--soc_version {soc_version} --sparsity 1 >> {scripts_name}")