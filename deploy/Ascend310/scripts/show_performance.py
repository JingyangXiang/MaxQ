import glob

msame_path = '~/project/tools/msame/out/main'

log_paths = glob.glob("../MaxQ/**/*.log", recursive=True)
for log_path in log_paths:
    with open(log_path, 'r') as f:
        content = f.readlines()
    if len(content) > 8 and content[-8].strip().endswith("ms"):
        speed = eval(content[-8].split(' ')[-2])
        print(f"{log_path}, {speed:.2f}")
