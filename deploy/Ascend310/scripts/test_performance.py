import glob
import os.path

msame_path = '~/project/tools/msame/out/main'

model_paths = glob.glob("../**/*.om", recursive=True)
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    dir_name = os.path.dirname(model_path)
    scripts = f"{msame_path} --loop 50 --model {model_path} > {os.path.join(dir_name, model_name.strip('om') + 'log')}"
    print(scripts)
    os.system(scripts)
