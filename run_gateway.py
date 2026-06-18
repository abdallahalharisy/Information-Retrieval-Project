"""Start the API Gateway (all SOA services on one port)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("gateway.app:app", host="127.0.0.1", port=8000, reload=False)
