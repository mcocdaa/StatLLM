#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-statllm}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "✓ StatLLM (${CONTAINER_NAME}) 已停止并清理"
