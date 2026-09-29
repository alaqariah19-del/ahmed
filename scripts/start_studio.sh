#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pkill -f "python app.py" 2>/dev/null || true
mkdir -p /tmp/private-photo-studio
nohup python app.py > /tmp/private-photo-studio/app.log 2>&1 &
for _ in $(seq 1 20); do
  if python - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:7860/api/health", timeout=1)
PY
  then
    echo "Private Photo Studio is running on http://127.0.0.1:7860"
    exit 0
  fi
  sleep 1
done
echo "Private Photo Studio did not start. Log:"
tail -n 80 /tmp/private-photo-studio/app.log || true
exit 1
