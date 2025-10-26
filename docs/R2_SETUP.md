# CloudFlare R2 配置指南

## 快速配置步骤

### 1. 获取 R2 凭证

1. 登录 [CloudFlare Dashboard](https://dash.cloudflare.com/)
2. 进入 **R2 Object Storage**
3. 创建一个 Bucket（例如：`papersys-data`）
4. 点击 **Manage R2 API Tokens**
5. 创建 API Token，获取：
   - **Access Key ID**
   - **Secret Access Key**（只显示一次，请保存）
6. 在右侧边栏找到你的 **Account ID**

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的凭证：

```bash
# R2 Access Key ID
AWS_ACCESS_KEY_ID=your_actual_access_key_id

# R2 Secret Access Key
AWS_SECRET_ACCESS_KEY=your_actual_secret_access_key

# R2 Endpoint (替换 your_account_id)
AWS_ENDPOINT=https://your_account_id.r2.cloudflarestorage.com

# R2 Region (固定值)
AWS_DEFAULT_REGION=auto
```

### 3. 修改 config.toml

编辑 `config.toml`，将数据库 URI 改为 R2 路径：

```toml
[database]
uri = "s3://your-bucket-name/papersys"
```

例如，如果你的 bucket 名为 `papersys-data`：

```toml
[database]
uri = "s3://papersys-data/papersys"
```

### 4. 验证配置

运行测试脚本验证连接：

```bash
uv run python -c "
from papersys.config import load_config, AppConfig
from papersys.database.manager import PaperManager
from pathlib import Path

config = load_config(AppConfig, Path('config.toml'))
print(f'Database URI: {config.database.uri}')

manager = PaperManager(uri=config.database.uri)
print('✅ 成功连接到 R2!')
"
```

## 工作原理

LanceDB 会自动从环境变量中读取 AWS S3 兼容的配置：

- `AWS_ACCESS_KEY_ID` - R2 访问密钥
- `AWS_SECRET_ACCESS_KEY` - R2 密钥
- `AWS_ENDPOINT` - R2 端点地址
- `AWS_DEFAULT_REGION` - 区域（R2 使用 "auto"）

当你在 `config.toml` 中使用 `s3://` 开头的 URI 时，LanceDB 会：
1. 识别这是 S3 兼容存储
2. 自动读取环境变量中的凭证
3. 使用指定的 endpoint 连接到 R2

## 本地开发 vs 生产环境

### 本地开发（使用本地存储）

```toml
[database]
uri = "data/papersys"
```

不需要设置环境变量，数据存储在本地 `data/papersys` 目录。

### 生产环境（使用 R2）

```toml
[database]
uri = "s3://papersys-data/papersys"
```

需要在 `.env` 中设置 R2 凭证。

## 常见问题

### Q: 如何在不同环境使用不同配置？

A: 可以使用多个配置文件：

```bash
# 本地开发
config.toml          # uri = "data/papersys"

# 生产环境
config.prod.toml     # uri = "s3://papersys-data/papersys"
```

然后在代码中选择性加载：

```python
import os
config_file = "config.prod.toml" if os.getenv("ENV") == "production" else "config.toml"
config = load_config(AppConfig, Path(config_file))
```

### Q: 可以使用其他 S3 兼容存储吗？

A: 可以！只需要修改 `AWS_ENDPOINT` 和 `AWS_DEFAULT_REGION`：

**MinIO:**
```bash
AWS_ENDPOINT=http://localhost:9000
AWS_DEFAULT_REGION=us-east-1
```

**AWS S3:**
```toml
uri = "s3://bucket-name/path"
```
环境变量只需要设置 `AWS_ACCESS_KEY_ID` 和 `AWS_SECRET_ACCESS_KEY`。

### Q: 如何测试连接？

A: 使用上面的验证配置脚本，或者运行：

```bash
uv run papersys init --help
```

如果能正常显示帮助信息且没有连接错误，说明配置正确。

## 安全提示

⚠️ **重要：**
- `.env` 文件已在 `.gitignore` 中，不会被提交到 Git
- 不要将凭证硬编码在代码中
- 不要将 `.env` 文件分享给他人
- API Token 只在创建时显示一次，请妥善保存

## 性能优化

使用 R2 时的性能建议：

1. **网络延迟**：R2 有全球 CDN，但首次请求可能较慢
2. **批量操作**：尽量使用批量读写减少请求次数
3. **缓存**：考虑在本地缓存热数据
4. **超时设置**：根据网络情况调整 `TIMEOUT` 和 `CONNECT_TIMEOUT`

## 成本估算

CloudFlare R2 定价（2025）：
- **存储**：$0.015/GB/月
- **写入**：免费
- **读取**：免费（无出站流量费用！）

这使得 R2 比 AWS S3 便宜很多，特别适合大量读取的场景。
