mkdir ../SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim
atc --model ../SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../SR-STE/2-4/batch_size-1-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../SR-STE/2-4/batch_size-2-N-2-M-4-resnet50-sim
atc --model ../SR-STE/2-4/batch_size-2-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../SR-STE/2-4/batch_size-2-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../SR-STE/2-4/batch_size-4-N-2-M-4-resnet50-sim
atc --model ../SR-STE/2-4/batch_size-4-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../SR-STE/2-4/batch_size-4-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../SR-STE/2-4/batch_size-8-N-2-M-4-resnet50-sim
atc --model ../SR-STE/2-4/batch_size-8-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../SR-STE/2-4/batch_size-8-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../SR-STE/2-4/batch_size-16-N-2-M-4-resnet50-sim
atc --model ../SR-STE/2-4/batch_size-16-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../SR-STE/2-4/batch_size-16-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim
atc --model ../MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../MaxQ/2-4/batch_size-1-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../MaxQ/2-4/batch_size-2-N-2-M-4-resnet50-sim
atc --model ../MaxQ/2-4/batch_size-2-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../MaxQ/2-4/batch_size-2-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../MaxQ/2-4/batch_size-4-N-2-M-4-resnet50-sim
atc --model ../MaxQ/2-4/batch_size-4-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../MaxQ/2-4/batch_size-4-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../MaxQ/2-4/batch_size-8-N-2-M-4-resnet50-sim
atc --model ../MaxQ/2-4/batch_size-8-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../MaxQ/2-4/batch_size-8-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
mkdir ../MaxQ/2-4/batch_size-16-N-2-M-4-resnet50-sim
atc --model ../MaxQ/2-4/batch_size-16-N-2-M-4-resnet50-sim.onnx --framework 5 --output ../MaxQ/2-4/batch_size-16-N-2-M-4-resnet50-sim/resnet50_fp16 --soc_version Ascend310B4
