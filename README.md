```bash
uv run marker --layout_batch_size 2 --recognition_batch_size 16 --disable_ocr --output_dir ./md ./pdf
```

## Notion sync

Push the latest summary snapshot to Notion once `summaries/last.jsonl` is up to date:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run papersys notion-sync \
  --database https://www.notion.so/29981ffdcde3804d8378c726c9eef5f1 \
  --snapshot data/git_store/<hash>/summaries/last.jsonl
```

Set `NOTION_TOKEN` in `.env` before running. Use `--dry-run` to inspect the batch without consuming API quota.
