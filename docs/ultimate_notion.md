# Ultimate Notion 快速上手

这份文档整理了团队目前用到的核心工作流，帮助你绕开官方文档里“写得又多又散”的部分。重点就是：如何用一个 token 连接 Notion、建立/复用会话、读取页面、修改内容，以及做一点点诊断。

## 1. 准备工作

- **创建 Notion 集成**：在 Notion 桌面或网页版里通过 `Settings → Connections → Develop or manage integrations` 新建一个 Internal 集成。
- **获取 token**：集成配置页的 *Internal Integration Secret* 会给你一个 `ntn_` 前缀的 token。
- **给页面授权**：在需要访问的页面右上角 `··· → Connections`，把刚才的集成加进去。父页面授权会自动下放到子页面。
- **本地环境变量**：把 token 写进 `.env`（或 shell 配置）并命名为 `NOTION_TOKEN`：
  ```bash
  NOTION_TOKEN="ntn_xxx"
  ```
  Ultimate Notion 默认会通过 `${env:NOTION_TOKEN}` 读取它。

## 2. 会话管理

`ultimate_notion.Session` 会缓存一个 Notion SDK client。常见用法有两种：

```python
import ultimate_notion as uno

# 方式 A：上下文管理器
with uno.Session() as notion:
    page = notion.get_page(page_id)

# 方式 B：全局复用，配合 get_or_create
otion = uno.Session.get_or_create()
page = notion.get_page(page_id)
# ...完成后想手动释放
notion.close()
```

推荐在 Notebook 里使用 **方式 B**，避免反复创建连接。`get_or_create()` 会返回已经存在的活动会话，因此不同单元格可以共享同一个 `Session`。

## 3. 页面与数据库

### 3.1 查找页面

```python
from ultimate_notion.utils import uuid_from_any

notion = uno.Session.get_or_create()
page_id = uuid_from_any("https://www.notion.so/Draft-27e81ffdcde380309eb3d9010ce98e4a")
page = notion.get_page(page_id)
```

如果只知道标题，可以用 `search_page`：

```python
result = notion.search_page("Draft")
page = result.item()  # SList: 如果有多个匹配，可以遍历或用 .first()/.last()
```

### 3.2 展示内容

- `page.show()`：在 Jupyter 环境里渲染富文本。
- `page.to_markdown()`：拿到纯 Markdown 文本（默认不包含嵌套 block）。
- `page.to_html()`：获取 HTML 字符串。

### 3.3 读取/修改属性

数据库页面的属性通过 `page.props` 访问：

```python
props = page.props
print(props.Status, props.owner)

props.Status = "In Progress"  # 会实时写回到 Notion
props["Owner"] = "lyk"
```

普通页面（非数据库条目）可以直接修改标题：

```python
page.title = "新标题"
page.reload()  # 如果需要刷新缓存
```

### 3.4 添加内容块

`ultimate_notion` 在 `ultimate_notion.blocks` 下提供了常见 block 工厂：

```python
from ultimate_notion import blocks

page.append(
    blocks.Heading1("今天的工作"),
    blocks.BulletedList(["整理 Draft 页面", "编写脚本"]),
)
```

## 4. 数据库操作

```python
from ultimate_notion import schema

# 创建数据库
parent = notion.get_page(parent_page_id)
db = notion.create_db(
    parent=parent,
    schema=schema.Database(
        db_title="任务追踪",
        properties={
            "Name": schema.Title(),
            "Status": schema.Select(options=["Todo", "Doing", "Done"]),
        },
    ),
)

# 新建条目
db.create_page(title="写 Ultimate Notion 手册", props={"Status": "Doing"})

# 查询条目
for row in db.iter_rows():
    print(row.props.Name, row.props.Status)
```

## 5. 常见问题排查

- **`RuntimeError: NOTION_TOKEN is missing`**：确认 `.env` 已加载（Notebook 里手动调用 `load_dotenv`），或直接在 shell 里 `export NOTION_TOKEN=...`。
- **`HTTPError: 401`**：token 填错或集成没授权页面。
- **页面/属性读取不到**：调用 `page.reload()` 或 `notion.cache.clear()`，防止旧缓存。
- **过多 Session**：统一使用 `Session.get_or_create()`，在清理时调用 `Session.get_active().close()`。

## 6. 推荐的最小脚本模板

```python
import os
from dotenv import load_dotenv
import ultimate_notion as uno
from ultimate_notion.utils import uuid_from_any

load_dotenv()  # 保证读取 NOTION_TOKEN

notion = uno.Session.get_or_create()
page = notion.get_page(uuid_from_any("https://www.notion.so/..."))
print(page.to_markdown())
```

把这个脚本作为“烟雾测试”，可以快速验证新环境或新 token 是否生效。

---

如果后续需要补充更高级的内容（同步器、数据库 schema 模板化等），可以在这个文档基础上继续扩展。