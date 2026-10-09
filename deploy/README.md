# StatLLM 镜像独立部署

适用于服务器只运行 Docker 容器、无需克隆完整源码的极简运维场景。
参考 `termflow` 与 `siteflow` 标准运维架构，使用 `ghcr.io/mcocdaa/statllm` 发布的官方多架构容器镜像。

## 目录结构
```text
/root/workspace/statllm/
├── .env.example          # 环境变量配置模板
├── .env                  # 运行期环境配置（权限 600）
├── init.sh               # 一键初始化/启动/更新脚本 (可执行)
├── stop.sh               # 停止并删除容器脚本 (可执行)
├── nginx.conf.example    # Nginx 反向代理配置参考
├── data/                 # 持久化数据库挂载目录
│   └── statllm.db        # SQLite 离散指纹基准数据库 (1,190 条样本)
└── README.md             # 本说明文档
```

## 快速启动

```bash
cd /root/workspace/statllm

# 1. 复制环境配置（可按需修改端口与版本标签）
cp .env.example .env
chmod 600 .env

# 2. 一键启动服务（自动拉取镜像并挂载持久化数据）
./init.sh

# 3. 查看容器运行状态
docker ps --filter "name=statllm"
curl -s http://127.0.0.1:8008/api/stats | jq .
```

## 停止服务

```bash
./stop.sh
```

## 版本升级

如需升级到最新版本或指定版本：
```bash
TAG=v0.2.0 ./init.sh
```
数据存储于宿主机 `./data/statllm.db`，升级或重建容器不会丢失任何已收集的真实样本。

## 反向代理配置

反向代理域名 `statllm.mcocdaa-newapi.xin` 指向 `http://127.0.0.1:8008`，详细配置参见 `nginx.conf.example`。
在宝塔面板添加反代站点时，目标 URL 填写 `http://127.0.0.1:8008`，发送域名保持 `$host`。
