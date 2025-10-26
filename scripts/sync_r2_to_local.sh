#!/usr/bin/env bash

# 同步 CloudFlare R2 数据到本地的脚本
# 使用方法: ./scripts/sync_r2_to_local.sh

set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# .env 文件路径
ENV_FILE="$PROJECT_ROOT/.env"

# 检查 .env 文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    echo "错误: 找不到 .env 文件: $ENV_FILE"
    exit 1
fi

# 检查 rclone 是否安装
if ! command -v rclone &> /dev/null; then
    echo "错误: 未找到 rclone 命令"
    echo "请先安装 rclone: https://rclone.org/install/"
    exit 1
fi

echo "正在从 .env 文件读取配置..."

# 读取 .env 文件中的配置
# 使用 grep 和 sed 提取环境变量
AWS_ACCESS_KEY_ID=$(grep -E "^AWS_ACCESS_KEY_ID=" "$ENV_FILE" | sed 's/AWS_ACCESS_KEY_ID=//' | tr -d '\r\n' | xargs)
AWS_SECRET_ACCESS_KEY=$(grep -E "^AWS_SECRET_ACCESS_KEY=" "$ENV_FILE" | sed 's/AWS_SECRET_ACCESS_KEY=//' | tr -d '\r\n' | xargs)
AWS_ENDPOINT=$(grep -E "^AWS_ENDPOINT=" "$ENV_FILE" | sed 's/AWS_ENDPOINT=//' | tr -d '\r\n' | xargs)
AWS_DEFAULT_REGION=$(grep -E "^AWS_DEFAULT_REGION=" "$ENV_FILE" | sed 's/AWS_DEFAULT_REGION=//' | tr -d '\r\n' | xargs)

# 验证必需的配置是否存在
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_ENDPOINT" ]; then
    echo "错误: .env 文件中缺少必需的配置项"
    echo "请确保以下配置项存在:"
    echo "  - AWS_ACCESS_KEY_ID"
    echo "  - AWS_SECRET_ACCESS_KEY"
    echo "  - AWS_ENDPOINT"
    exit 1
fi

# 如果未设置 region，使用默认值
if [ -z "$AWS_DEFAULT_REGION" ]; then
    AWS_DEFAULT_REGION="auto"
fi

# 读取 config.toml 中的 bucket 和路径配置
CONFIG_FILE="$PROJECT_ROOT/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到 config.toml 文件: $CONFIG_FILE"
    exit 1
fi

# 从 config.toml 中提取 database.uri
# 格式: uri = "s3://bucket-name/path"
# 优先查找未注释的 S3 URI，如果没有则查找注释的
DATABASE_URI="s3://papersys/lancedb"

if [ -z "$DATABASE_URI" ]; then
    echo "错误: 无法从 config.toml 中读取 S3 格式的 database.uri"
    echo "请确保 config.toml 中有 s3:// 格式的配置（可以是注释的）"
    exit 1
fi

# 解析 S3 URI
if [[ $DATABASE_URI =~ ^s3://([^/]+)/(.+)$ ]]; then
    BUCKET_NAME="${BASH_REMATCH[1]}"
    REMOTE_PATH="${BASH_REMATCH[2]}"
else
    echo "错误: database.uri 格式不正确，期望格式: s3://bucket-name/path"
    echo "当前值: $DATABASE_URI"
    exit 1
fi

# 本地目标路径
LOCAL_PATH="$PROJECT_ROOT/data/lancedb"

echo ""
echo "配置信息:"
echo "  Bucket: $BUCKET_NAME"
echo "  远程路径: $REMOTE_PATH"
echo "  本地路径: $LOCAL_PATH"
echo "  Endpoint: $AWS_ENDPOINT"
echo ""

# 创建本地目录（如果不存在）
mkdir -p "$LOCAL_PATH"

# 配置 rclone remote（如果不存在）
REMOTE_NAME="papersys-r2"

echo "检查 rclone 配置..."
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
    echo "创建 rclone remote: $REMOTE_NAME"
    rclone config create "$REMOTE_NAME" s3 \
        provider=Cloudflare \
        access_key_id="$AWS_ACCESS_KEY_ID" \
        secret_access_key="$AWS_SECRET_ACCESS_KEY" \
        endpoint="$AWS_ENDPOINT" \
        region="$AWS_DEFAULT_REGION" \
        acl=private
    echo "rclone remote 创建成功"
else
    echo "rclone remote '$REMOTE_NAME' 已存在"
fi

echo ""
echo "开始同步数据..."
echo "从: ${REMOTE_NAME}:${BUCKET_NAME}/${REMOTE_PATH}"
echo "到: $LOCAL_PATH"
echo ""

# 使用 rclone sync 同步数据
# --progress: 显示进度
# --verbose: 显示详细信息
# --transfers: 并行传输数量
# --checkers: 并行检查数量
rclone sync \
    "${REMOTE_NAME}:${BUCKET_NAME}/${REMOTE_PATH}" \
    "$LOCAL_PATH" \
    --progress \
    --verbose \
    --transfers 4 \
    --checkers 8

echo ""
echo "同步完成！"
echo "数据已同步到: $LOCAL_PATH"
