#!/usr/bin/env bash
# StatLLM 镜像部署脚本（无需克隆源码）
# 参考 termflow / siteflow 架构规范
# 用法: TAG=v0.1.0 STATLLM_PORT=8008 ./init.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. 确保环境配置文件存在
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "[init] 已从 .env.example 创建 .env"
fi

# 加载 .env
set -a
source .env
set +a

TAG="${TAG:-v0.1.0}"
PORT="${STATLLM_PORT:-8008}"
BIND_HOST="${STATLLM_BIND_HOST:-127.0.0.1}"
CONTAINER_NAME="${CONTAINER_NAME:-statllm}"
IMAGE="ghcr.io/mcocdaa/statllm:${TAG}"

echo "[init] 正在拉取镜像: ${IMAGE} ..."
docker pull "$IMAGE"

# 2. 准备持久化数据目录与基准指纹库
mkdir -p data
if [[ ! -f data/statllm.db ]]; then
    echo "[init] 初始化持久化数据库，提取内置 15 款大模型基准指纹 (1,190 条全真实样本)..."
    docker run --rm --entrypoint cat "$IMAGE" /app/statllm.db > data/statllm.db
    chmod 0644 data/statllm.db
fi

# 3. 停止并清理旧容器
echo "[init] 重启容器: ${CONTAINER_NAME} ..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# 4. 运行新容器
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${BIND_HOST}:${PORT}:8000" \
    -v "$PWD/data/statllm.db:/app/statllm.db" \
    --env-file .env \
    "$IMAGE"

# 5. 健康检查
echo "[init] 等待服务就绪并执行健康检查..."
sleep 2
MAX_RETRIES=10
COUNT=0
while [[ $COUNT -lt $MAX_RETRIES ]]; do
    if curl -s -f "http://${BIND_HOST}:${PORT}/api/stats" >/dev/null 2>&1; then
        echo "✓ StatLLM 已成功启动并就绪: http://${BIND_HOST}:${PORT}"
        echo "✓ API 统计端点健康: http://${BIND_HOST}:${PORT}/api/stats"
        exit 0
    fi
    COUNT=$((COUNT+1))
    sleep 1
done

echo "[warning] 服务已启动，正在后台初始化，可通过 'docker logs ${CONTAINER_NAME}' 查看日志"
