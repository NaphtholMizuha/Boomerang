from pathlib import Path
import subprocess as sp

directory = Path("config/batch")

file_names = [f.stem for f in directory.iterdir() if f.is_file()]


for file_name in file_names:
    command = f"/home/wuzihou/miniforge3/envs/fl/bin/python run.py -c config/batch/{file_name}.toml &> log/{file_name}.log"
    print(command)
    sp.run(["screen", "-dmS", file_name, "bash", "-c", command])
