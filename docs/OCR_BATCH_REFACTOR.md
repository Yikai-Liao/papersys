# OCR Batch Processing 重构文档

## 背景

原有实现直接使用 arXiv PDF URL 调用 Mistral OCR API，但 Mistral 对 arXiv 访问没有做爬虫优化，容易触发 403 错误或人机验证。

## 解决方案

重构为三阶段流程：
1. **下载阶段**：从 arXiv 本地下载 PDF（带频率限制）
2. **上传阶段**：上传 PDF 到 Mistral Cloud
3. **批处理阶段**：使用 Batch API 处理 OCR

## 关键改进

### 1. arXiv 访问频率限制

参考 `reference/ArxivEmbedding/src/oai.py` 的实现，添加访问频率控制：

```python
# 每次请求之间等待 3 秒，避免触发 403
ARXIV_DOWNLOAD_DELAY = 3.0
```

- 请求间隔：3 秒（下载 100 篇论文约需 5 分钟）
- User-Agent：使用友好的标识
- 重试机制：遇到 403 时加倍等待时间
- 最大重试：3 次

### 2. PDF 缓存管理

```python
pdf_cache_dir = pathlib.Path(tempfile.gettempdir()) / "papersys_ocr_cache"
```

- 默认缓存到系统临时目录
- 可自定义缓存目录（推荐 `data/pdf_cache/`）
- 支持复用已下载的 PDF
- 可选择处理后清理（`cleanup_pdfs=True`）

### 3. Mistral Cloud 文件管理

完整的文件生命周期管理：

1. **上传 PDF**：`client.files.upload(purpose="ocr")`
2. **获取签名 URL**：`client.files.get_signed_url()`
3. **批处理 OCR**：使用签名 URL 创建 batch job
4. **删除文件**：处理完成后从 Mistral Cloud 删除

### 4. API 使用

#### 单篇处理

```python
from papersys.ocr import ocr_by_id

response = ocr_by_id(
    arxiv_id="2201.04234",
    pdf_cache_dir=pathlib.Path("data/pdf_cache"),  # 可选
    cleanup_pdf=False  # 是否删除 PDF
)
```

#### 批量处理

```python
from papersys.ocr import ocr_by_id_batch

results = ocr_by_id_batch(
    arxiv_ids=["2201.04234", "2410.12613", "2505.11739"],
    pdf_cache_dir=pathlib.Path("data/pdf_cache"),  # 可选
    cleanup_pdfs=False,  # 是否删除 PDF
    wait_for_completion=True,  # 是否等待完成
    poll_interval=10  # 状态检查间隔（秒）
)

# results: dict[str, OCRResponse]
for arxiv_id, ocr_response in results.items():
    print(f"Processed {arxiv_id}: {len(ocr_response.pages)} pages")
```

## 时间估算

### 下载阶段
- 单篇耗时：3 秒（频率限制） + 下载时间（1-5 秒）
- 100 篇论文：约 5-8 分钟

### 上传阶段
- 单篇耗时：1-3 秒
- 100 篇论文：约 2-5 分钟

### OCR 处理阶段
- Batch API 并行处理
- 时间取决于最长的文档
- 估算：5-20 分钟（取决于页数和复杂度）

**总计**：100 篇论文约 12-33 分钟

## 错误处理

### 下载失败
- 单篇下载失败不会中断整个批次
- 失败的论文会被跳过并记录警告
- 重试 3 次后放弃

### 上传失败
- 上传失败会抛出异常（需要 API key 有效）

### OCR 失败
- Batch job 失败会抛出异常
- 包含详细的失败信息

## 成本优化

1. **PDF 缓存**：避免重复下载相同论文
2. **Batch API**：比单篇请求成本更低
3. **文件清理**：及时删除 Mistral Cloud 文件（避免存储费用）

## 测试

```bash
# 运行测试脚本
uv run python scripts/test_ocr_batch.py
```

测试脚本会：
1. 下载 3 篇示例论文
2. 批量处理 OCR
3. 保存为 Markdown + 图片
4. 输出到 `data/ocr_responses/`

## 兼容性

- ✅ `response2md()` 函数完全兼容新旧 API 返回格式
- ✅ 支持对象和字典两种响应格式
- ✅ 现有代码无需修改

## 配置建议

### 开发环境
```python
pdf_cache_dir = pathlib.Path("data/pdf_cache")  # 缓存到项目目录
cleanup_pdfs = False  # 保留 PDF 用于调试
```

### 生产环境
```python
pdf_cache_dir = pathlib.Path("/tmp/papersys_ocr_cache")  # 使用临时目录
cleanup_pdfs = True  # 自动清理节省空间
```

## 监控建议

关键日志：
- `[INFO] Downloading N PDFs from arXiv...`
- `[INFO] Successfully prepared N PDFs`
- `[INFO] Uploading N PDFs to Mistral Cloud...`
- `[SUCCESS] Batch job created: {job_id}`
- `[DEBUG] Batch status: {status}`
- `[SUCCESS] Batch processing complete! Processed N papers.`

错误日志：
- `[WARNING] Download failed for {arxiv_id}: HTTP {status_code}`
- `[ERROR] Failed to download {arxiv_id} after N attempts`
- `[WARNING] Failed to delete file {file_id}: {error}`

## 未来优化方向

1. **并行下载**：使用 `asyncio` 提升下载速度（需保持频率限制）
2. **断点续传**：支持大型 batch job 的中断恢复
3. **智能重试**：根据 HTTP 状态码调整重试策略
4. **进度条**：添加 tqdm 进度显示
5. **元数据缓存**：记录已处理论文，避免重复 OCR
