#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./venv/bin/uvicorn app.main:app --app-dir services/fastapi_service --host 127.0.0.1 --port 9000
