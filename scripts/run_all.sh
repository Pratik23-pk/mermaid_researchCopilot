#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$0")/.."
cd "$ROOT"

./scripts/run_fastapi.sh &
FASTAPI_PID=$!

cleanup() {
  kill "$FASTAPI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

./scripts/run_django.sh
