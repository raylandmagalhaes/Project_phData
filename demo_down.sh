#!/usr/bin/env bash
set -euo pipefail
echo "Stopping demo stack…"
docker compose down
echo "Stopped."
