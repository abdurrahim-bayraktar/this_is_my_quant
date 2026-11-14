"""
Dashboard launcher — starts both backend (FastAPI) and frontend (Vite) servers.

Usage:
    python dashboard/start.py
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
PROJECT_ROOT = ROOT.parent


def main():
    procs = []

    try:
        # Start FastAPI backend
        print("[BACKEND] Starting FastAPI backend on http://localhost:8000 ...")
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "dashboard.backend.main:app",
             "--reload", "--port", "8000", "--host", "0.0.0.0"],
            cwd=str(PROJECT_ROOT),
        )
        procs.append(backend_proc)
        time.sleep(2)

        # Start Vite frontend
        print("[FRONTEND] Starting Vite frontend on http://localhost:5173 ...")
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            shell=True,
        )
        procs.append(frontend_proc)

        print("\n" + "=" * 60)
        print("  Dashboard is running!")
        print("  Frontend:  http://localhost:5173")
        print("  Backend:   http://localhost:8000")
        print("  API Docs:  http://localhost:8000/docs")
        print("=" * 60)
        print("\nPress Ctrl+C to stop both servers.\n")

        # Wait for either to exit
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"Process {p.pid} exited with code {p.returncode}")
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("Done.")


if __name__ == "__main__":
    main()
