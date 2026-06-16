import subprocess
import sys
import os

scripts = [
    os.path.join("src", "portion1_data.py"),
    os.path.join("src", "portion2_perceptron.py"),
    os.path.join("src", "portion3_delta.py"),
    os.path.join("src", "portion4_activations.py"),
    os.path.join("src", "portion5_lr_tuning.py"),
    os.path.join("src", "portion6_comparison.py"),
]

for script in scripts:
    print(f"\n{'=' * 50}")
    print(f"  Running: {script}")
    print(f"{'=' * 50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nError in {script}. Stopping.")
        sys.exit(1)
    print(f"  Done: {script}")

print("\nAll portions finished. Check the outputs/ folder for plots.")
