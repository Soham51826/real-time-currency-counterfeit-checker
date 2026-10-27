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
    
    # Priority: If .venv exists, use its Python interpreter (Windows or Unix)
    venv_win = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    venv_unix = os.path.join(script_dir, ".venv", "bin", "python")
    
    if os.path.isfile(venv_win):
        python_bin = venv_win
    elif os.path.isfile(venv_unix):
        python_bin = venv_unix
    else:
        python_bin = sys.executable
    
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
