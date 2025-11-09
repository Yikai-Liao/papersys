# Notion API 使用指南与最佳实践

本文档记录了在使用 `ultimate-notion` 库与 Notion API 交互时的调研过程、遇到的问题以及最佳实践。

## 目录

- [问题背景](#问题背景)
- [调研过程](#调研过程)
- [最佳实践](#最佳实践)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

## 问题背景

在实现 `papersys notion-sync` 功能时，需要将论文摘要数据同步到 Notion 数据库。初始实现遇到了以下问题：

1. **502 Bad Gateway 错误**：Notion API 间歇性返回 502 错误
2. **属性缺失**：部分字段（如 `id`）没有被成功上传
3. **性能问题**：每个页面需要 4-5 次 API 调用，效率低下
4. **代码复杂**：同时处理新建和更新逻辑，代码冗余

## 调研过程

### 1. ultimate-notion 的层次结构

`ultimate-notion` 是对 Notion API 的高级封装，包含三个层次：

```
ultimate_notion (高级接口)
    ├── Session, Database, Page, Block (封装类)
    ├── obj_api (中级对象API)
    │   ├── endpoints.py (API端点封装)
    │   ├── blocks.py (底层对象定义)
    │   └── props.py (属性类型定义)
    └── notion_client (底层SDK)
```

**关键发现**：
- `ultimate_notion.database.Database` ≠ `ultimate_notion.obj_api.blocks.Database`
- 高级接口不支持所有底层 API 参数
- 需要使用 `.obj_ref` 在不同层次间转换

### 2. session.create_page() 的限制

查看 `session.py` 源码发现：

```python
def create_page(
    self, parent: Page | Database, title: Text | str | None = None, blocks: Sequence[Block] | None = None
) -> Page:
    """Create a new page in a `parent` page or database with a given `title`."""
    title_obj = title if title is None else Title(title).obj_ref
    # We don't use the `children` parameter as we would need to call `list` afterwards...
    page = Page.wrap_obj_ref(self.api.pages.create(parent=parent.obj_ref, title=title_obj))
    self.cache[page.id] = page

    if blocks:
        blocks_iter = _chunk_blocks_for_api(page, blocks)
        _append_block_chunks(blocks_iter)

    return page
```

**问题**：
- ❌ 只支持 `title` 和 `blocks` 参数
- ❌ 不支持传入其他 properties（如 `id`、`authors`、`dates` 等）
- ❌ 需要额外调用 `page.props[xxx] = yyy` 设置属性

### 3. api.pages.create() 的完整功能

深入 `obj_api/endpoints.py` 发现底层 API 支持更多参数：

```python
# From: .venv/lib/python3.12/site-packages/ultimate_notion/obj_api/endpoints.py
def create(
    self,
    parent: ParentRef | Page | Database,
    title: Title | None = None,
    properties: dict[str, PropertyValue] | None = None,  # ✅ 支持！
    children: list[Block] | None = None,                # ✅ 支持！
) -> Page:
    """Add a page to the given parent (Page or Database)."""
    if parent is None:
        msg = "'parent' must be provided"
        raise ValueError(msg)

    match parent:
        case Page():
            parent = PageRef.build(parent)
            parent_id = parent.page_id
        case Database():
            parent = DatabaseRef.build(parent)
            parent_id = parent.database_id
        case _:
            msg = f'Unsupported parent of type {type(parent)}'
            raise ValueError(msg)

    request: dict[str, Any] = {'parent': parent.serialize_for_api()}

    # the API requires a properties object, even if empty
    if properties is None:
        properties = {}

    if title is not None:
        properties['title'] = title

    request['properties'] = {
        name: prop.serialize_for_api() if prop is not None else None 
        for name, prop in properties.items()
    }

    if children is not None:
        request['children'] = [child.serialize_for_api() for child in children if child is not None]

    _logger.debug(f'Creating new page below page with id `{parent_id}`.')
    data = self.raw_api.create(**request)
    return Page.model_validate(data)
```

**关键发现**：
- ✅ 支持同时传入 `properties` 和 `children`
- ✅ 一次 API 调用即可创建带所有属性和内容的页面
- ✅ 性能最优方案

## 最佳实践

### 1. 使用底层 API 一次性创建页面

**推荐做法**：

```python
from ultimate_notion.props import Title, Text, Date, Number, MultiSelect, PropertyValue

def _prepare_properties(record: dict, database: Database) -> dict[str, PropertyValue]:
    """准备所有属性"""
    properties = {}
    
    # Title 字段（数据库的标题列）
    properties["title"] = Title(record.get("paper_title") or record.get("id") or "")
    
    # Text 字段
    properties["id"] = Text(record.get("id"))
    properties["authors"] = Text(record.get("authors"))
    
    # MultiSelect 字段
    if keywords := record.get("keywords"):
        properties["keywords"] = MultiSelect(keywords.split(","))
    
    # Date 字段
    if publish_date := record.get("publish_date"):
        properties["publish_date"] = Date(publish_date)
    
    # Number 字段
    if score := record.get("score"):
        properties["score"] = Number(score)
    
    return properties


def _create_page_with_properties(
    session: uno.Session,
    database: Database,
    properties: dict[str, PropertyValue],
    blocks: list[uno.Block] | None,
) -> Page:
    """一次 API 调用创建带所有属性和内容的页面"""
    # 转换为 obj_ref 格式
    properties_obj = {name: prop.obj_ref for name, prop in properties.items()}
    blocks_obj = [block.obj_ref for block in blocks] if blocks else None
    
    # 使用底层 API（注意：必须传 database.obj_ref）
    page_obj = session.api.pages.create(
        parent=database.obj_ref,  # ⚠️ 必须用 .obj_ref
        properties=properties_obj,
        children=blocks_obj,
    )
    
    # 包装并缓存
    page = Page.wrap_obj_ref(page_obj)
    session.cache[page.id] = page
    
    return page
```

**关键要点**：
1. 使用 `session.api.pages.create` 而不是 `session.create_page`
2. 传入 `database.obj_ref` 而不是 `database` 本身
3. 所有属性都通过 `properties` 参数传入
4. 所有内容块通过 `children` 参数传入
5. 记得将 `PropertyValue` 和 `Block` 对象转换为 `.obj_ref` 格式

### 2. 属性类型映射

| 数据类型 | Notion Schema | PropertyValue 类型 | 示例 |
|---------|---------------|-------------------|------|
| 文本 | `notion_schema.Text()` | `Text(value)` | `Text("ArXiv ID")` |
| 标题 | `notion_schema.Title()` | `Title(value)` | `Title("论文标题")` |
| 数字 | `notion_schema.Number()` | `Number(value)` | `Number(4.5)` |
| 日期 | `notion_schema.Date()` | `Date(value)` | `Date("2024-01-01")` |
| 单选 | `notion_schema.Select()` | `Select(value)` | `Select("like")` |
| 多选 | `notion_schema.MultiSelect()` | `MultiSelect([...])` | `MultiSelect(["AI", "ML"])` |

### 3. 错误处理与重试

Notion API 会间歇性返回 502/503 错误，需要实现重试机制：

```python
import time
from loguru import logger

MAX_RETRIES = 3
RETRY_DELAY = 2.0
RETRY_BACKOFF = 2.0

def retry_on_502(func, *args, **kwargs):
    """带指数退避的重试机制"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            is_retryable = any(
                indicator in error_msg 
                for indicator in ["502", "bad gateway", "503", "service unavailable", "timeout"]
            )
            
            if is_retryable:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                    logger.warning(
                        "Notion API error (attempt {}/{}): {} - Retrying in {:.1f}s...",
                        attempt + 1, MAX_RETRIES, str(e), delay
                    )
                    time.sleep(delay)
                    continue
            raise
    
    if last_exception:
        raise last_exception
```

### 4. Markdown 到 Notion Blocks 的转换

使用 `MarkdownToNotionConverter` 离线转换 Markdown：

```python
from papersys.notion.md2notion import MarkdownToNotionConverter

converter = MarkdownToNotionConverter(session=session)

def _prepare_page_blocks(record: dict, converter: MarkdownToNotionConverter) -> list[uno.Block]:
    """离线准备所有 blocks"""
    sections = [
        ("One-sentence Summary", record.get("one_sentence_summary")),
        ("Problem Background", record.get("problem_background")),
        ("Method", record.get("method")),
        ("Experiment", record.get("experiment")),
    ]
    
    blocks = []
    for heading, content in sections:
        if not content:
            continue
        blocks.append(uno.Heading2(heading))
        section_blocks = converter.convert(str(content))
        blocks.extend(section_blocks)
        blocks.append(uno.Paragraph(""))  # 分隔符
    
    # 移除末尾空段落
    if blocks and isinstance(blocks[-1], uno.Paragraph) and not str(blocks[-1]).strip():
        blocks.pop()
    
    return blocks
```

### 5. 完整的同步流程

```python
def sync_snapshot_to_notion(snapshot_path: Path, database_ref: str, **kwargs) -> NotionSyncReport:
    """同步快照到 Notion"""
    session = uno.Session.get_or_create()
    database = retry_on_502(session.get_db, database_ref)
    retry_on_502(_ensure_schema, database)
    
    converter = MarkdownToNotionConverter(session=session)
    records = _load_snapshot(snapshot_path)
    
    for record in tqdm(records, desc="Creating pages in Notion", unit="paper"):
        # Step 1: 离线准备 blocks
        blocks = _prepare_page_blocks(record, converter)
        
        # Step 2: 准备所有 properties
        properties = _prepare_properties(record, database)
        
        # Step 3: 一次 API 调用创建页面（包含所有属性和内容）
        page = retry_on_502(_create_page_with_properties, session, database, properties, blocks)
    
    session.close()
```

## 性能对比

### 旧实现（4-5 次 API 调用/页面）

```python
# 1. 创建页面（只有 title）
page = database.create_page(title=page_title)

# 2. Append blocks
page.append(blocks)

# 3-N. 逐个设置属性（每个都是一次 API 调用）
page.props["id"] = record_id
page.props["authors"] = authors
page.props["publish_date"] = publish_date
# ... 更多属性
```

**性能**：
- 5个页面耗时：~2-3分钟
- 每页面：~24-36秒
- API 调用：20-25次

### 新实现（1 次 API 调用/页面）

```python
# 一次调用完成所有事情
page = session.api.pages.create(
    parent=database.obj_ref,
    properties=properties_obj,  # 所有属性
    children=blocks_obj,        # 所有内容
)
```

**性能**：
- 5个页面耗时：~4-5秒
- 每页面：~0.8-1.0秒
- API 调用：5次

**提升**：**30-40倍性能提升！** 🚀

## 常见问题

### Q1: 为什么不能直接用 `session.create_page()`？

A: `session.create_page()` 只支持 `title` 和 `blocks` 参数，不支持传入其他 properties。如果需要设置其他属性，必须事后通过 `page.props[xxx] = yyy` 设置，这会产生额外的 API 调用。

### Q2: 什么时候用 `.obj_ref`？

A: 当你需要在 `ultimate_notion` 的高级接口和底层 `obj_api` 之间转换时：

```python
# ✅ 正确
session.api.pages.create(parent=database.obj_ref, ...)
properties_obj = {name: prop.obj_ref for name, prop in properties.items()}

# ❌ 错误
session.api.pages.create(parent=database, ...)  # TypeError
```

### Q3: 如何处理 "Unsupported parent of type" 错误？

A: 确保传入 `.obj_ref`：

```python
# ❌ 错误
page_obj = session.api.pages.create(parent=database, ...)

# ✅ 正确
page_obj = session.api.pages.create(parent=database.obj_ref, ...)
```

### Q4: 如何处理 "xxx is expected to be yyy" 错误？

A: 这说明数据库中该字段的类型与你传入的类型不匹配。检查数据库 schema：

```python
# 如果 Notion 说 "institution is expected to be multi_select"
# ❌ 错误
properties["institution"] = Text("MIT")

# ✅ 正确
properties["institution"] = MultiSelect(["MIT"])
```

### Q5: 如何区分 title 字段和普通文本字段？

A: Notion 数据库中有且仅有一个 "title" 字段，这是特殊的：

```python
# Title 字段（数据库的标题列）
properties["title"] = Title("这是页面标题")

# 普通文本字段
properties["paper_title"] = Text("论文标题")  # 如果是文本类型
properties["id"] = Text("2501.12345")
```

### Q6: 502 错误如何处理？

A: 实现指数退避重试机制（见上文 "错误处理与重试" 章节）。Notion API 的 502 错误通常是暂时性的，重试 2-3 次基本能解决。

### Q7: 如何确保 blocks 正确转换？

A: 使用 `MarkdownToNotionConverter` 并注意：

1. 转换是离线的，不会触发 API 调用
2. 转换后需要调用 `.obj_ref` 获取底层对象
3. 某些嵌套 block 可能不被 API 支持，需要单独 append

```python
# ✅ 正确
blocks = converter.convert(markdown_text)
blocks_obj = [block.obj_ref for block in blocks]

# ❌ 错误
blocks_obj = [block for block in blocks]  # 缺少 .obj_ref
```

## 总结

通过深入研究 `ultimate-notion` 的源码和 Notion API 的底层接口，我们发现：

1. **使用底层 API**：`session.api.pages.create` 而不是 `session.create_page`
2. **一次性传入所有数据**：properties + children 一次搞定
3. **正确的类型转换**：使用 `.obj_ref` 在不同层次间转换
4. **合理的错误处理**：实现指数退避重试机制
5. **离线准备数据**：减少在线转换的时间

最终实现了 **30-40倍的性能提升**，同时代码更简洁、更易维护。

---

**参考文件**：
- `/papersys/notion/summary_sync.py` - 完整实现
- `/papersys/notion/md2notion.py` - Markdown 转换器
- `.venv/lib/python3.12/site-packages/ultimate_notion/` - ultimate-notion 源码

**更新日期**：2025-11-09
