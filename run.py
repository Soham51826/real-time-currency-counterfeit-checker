"""
CurrencyGuard AI - Launcher Entrypoint
Allows running the project via:
    python run.py
"""
import os
import sys
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, "app.py")
    
    # Priority: If .venv exists, use its Python interpreter
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.isfile(venv_python) else sys.executable
    
    cmd = [python_bin, "-m", "streamlit", "run", app_file] + sys.argv[1:]
    print("=" * 60)
    print("Launching CurrencyGuard AI Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Streamlit: {e}")

if __name__ == "__main__":
    main()
