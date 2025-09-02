#!/usr/bin/env bash
set -euo pipefail

# Generate a temporary, random SECRET_KEY for this run only (not printed)
if ! command -v python >/dev/null 2>&1; then
  echo "Python is required to generate a SECRET_KEY for the demo." >&2
  exit 1
fi

SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(64))')"

echo "Starting demo stack with a temporary SECRET_KEY"
SECRET_KEY="$SECRET_KEY" docker compose --env-file .env.demo up -d

echo "Done. Try:  curl -s http://localhost:8000/health"
