"""
Run each SOA service on its own port (independent deployment).

Ports:
  8001 preprocessing | 8002 indexing | 8003 retrieval
  8004 ranking | 8005 query-refinement | 8006 evaluation
  8000 gateway (use run_gateway.py instead for unified mode)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SERVICES = [
    ("preprocessing", 8001, "services.preprocessing.app:app"),
    ("indexing", 8002, "services.indexing.app:app"),
    ("retrieval", 8003, "services.retrieval.app:app"),
    ("ranking", 8004, "services.ranking.app:app"),
    ("query_refinement", 8005, "services.query_refinement.app:app"),
    ("evaluation", 8006, "services.evaluation.app:app"),
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_services.py <service_name|all>")
        print("Services:", ", ".join(s[0] for s in SERVICES))
        sys.exit(1)

    target = sys.argv[1].lower()
    to_run = SERVICES if target == "all" else [s for s in SERVICES if s[0] == target]

    if not to_run:
        print(f"Unknown service: {target}")
        sys.exit(1)

    procs = []
    for name, port, app_path in to_run:
        print(f"Starting {name} on port {port}...")
        p = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", app_path, "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(ROOT),
        )
        procs.append((name, p))

    print("\nServices running. Press Ctrl+C to stop.")
    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        for name, p in procs:
            p.terminate()
            print(f"Stopped {name}")


if __name__ == "__main__":
    main()
