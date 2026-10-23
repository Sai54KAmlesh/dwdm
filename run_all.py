import subprocess, sys

for script in ["generate_data.py", "build_dw.py", "olap.py", "mining.py"]:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, script], check=True)
if "--no-dashboard" not in sys.argv:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
