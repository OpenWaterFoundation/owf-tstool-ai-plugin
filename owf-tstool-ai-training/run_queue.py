import os
import subprocess

SCRIPT_DIR = "models_to_train"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all .py scripts
scripts = [f for f in os.listdir(SCRIPT_DIR) if f.endswith('.py')]

for script in scripts:
    name = os.path.splitext(script)[0]
    script_path = os.path.join(SCRIPT_DIR, script)
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_path, exist_ok=True)

    print(f"Running {script}...")

    log_file = os.path.join(out_path, 'execution.log')
    with open(log_file, 'w') as log:
        result = subprocess.run(
            ["python", script_path, "--output_dir", out_path],
            stdout=log,
            stderr=subprocess.STDOUT
        )

    print(f"{script} finished with exit code {result.returncode}")
