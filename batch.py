from pathlib import Path
import subprocess as sp
import sys
directory = Path("config/batch")

file_names = [f.stem for f in directory.iterdir() if f.is_file()]
python = sys.executable

for file_name in file_names:
    command = f"{python} run.py -c config/batch/{file_name}.toml &> log/{file_name}.log"
    print(command)
    sp.run(["screen", "-dmS", file_name, "bash", "-c", command])
