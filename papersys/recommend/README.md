# 推荐系统 - S3读取优化版本

## 核心优化策略

### 问题
原来的实现会先读取完整的embedding数据，然后再进行过滤和采样，导致大量不必要的S3读取。

### 解决方案
**先筛选ID，最后才读取embedding**，大幅减少S3读取次数和流量。

## 优化流程

### 1. 加载偏好数据 (`load_preference_data`)

**旧流程：**
```
读取完整preference表 -> 读取完整metadata -> 读取完整embedding -> 过滤
```

**新流程（优化后）：**
```
1. 读取preference表（只读ID和preference列）
2. 读取metadata（只读ID和categories列）
3. 在Polars中过滤类别
4. 检查哪些ID有embedding（只读embedding表的ID列）
5. 最后才读取过滤后ID的embedding向量 ✅
```

**优势：** 避免读取不需要的embedding数据

---

### 2. 加载背景数据 (`load_background_data`)

**旧流程：**
```
读取所有符合条件的embedding -> 在内存中采样
```

**新流程（优化后）：**
```
1. 读取embedding表（只读ID列）
2. 读取metadata（只读ID和categories列），应用时间过滤
3. 在Arrow/Polars中过滤类别
4. 找出 (有embedding ∩ 符合类别 ∩ 符合时间 - 已标注) 的ID集合
5. 【关键】在ID层面进行随机采样（例如从10万条中采样5000条）
6. 最后才读取采样后ID的embedding向量 ✅
```

**优势：** 
- 训练时只读取需要的5000条embedding，而不是10万条
- S3读取量减少 95%！

---

### 3. 预测阶段 (`predict`)

**旧流程：**
```
读取所有目标期间的embedding -> 预测
```

**新流程（优化后）：**
```
1. 用同样的方法筛选出目标期间的ID列表
2. 最后才读取这些ID的embedding向量 ✅
3. 预测和推荐
```

**优势：** 只读取需要预测的论文的embedding

---

## 性能对比示例

假设：
- metadata表有 100,000 条记录
- 符合类别的有 50,000 条
- 训练需要 5,000 条背景数据

### 旧方案
- 读取 50,000 条完整embedding
- 在内存中采样 5,000 条
- **S3读取：50,000 条 embedding**

### 新方案
- 读取 50,000 个ID（小数据）
- 采样 5,000 个ID
- 读取 5,000 条embedding
- **S3读取：5,000 条 embedding** ✅

**节省流量：90%**

---

## 使用示例

```python
from papersys.config import load_config
from papersys.database.manager import PaperManager
from papersys.recommend import Recommender

# 加载配置
config = load_config("config.toml")

# 连接数据库
manager = PaperManager(uri="data/papersys")

# 创建推荐器
recommender = Recommender(manager, config.recommend)

# 训练模型（自动优化S3读取）
recommender.fit(categories=config.paper.categories)

# 预测和推荐（自动优化S3读取）
results = recommender.predict(
    categories=config.paper.categories,
    last_n_days=7
)

# 查看推荐结果
recommended = results.filter(pl.col("show") == 1)
print(recommended)
```

## 配置说明

在 `config.toml` 中配置推荐参数：

```toml
[recommend]
neg_sample_ratio = 5.0  # 负样本比例
seed = 42

[recommend.logistic_regression]
C = 1.0
max_iter = 1000

[recommend.predict]
last_n_days = 7          # 预测最近N天
sample_rate = 0.15       # 推荐比例
high_threshold = 0.95    # 高分阈值
boundary_threshold = 0.5 # 边界阈值
```

## 技术细节

- 使用 **LanceDB** 的列式存储特性，只读取需要的列
- 使用 **PyArrow** 的 compute 模块进行高效的集合操作
- 使用 **Polars** 进行内存高效的数据处理
- 采样在 **ID层面** 完成，避免读取大量embedding数据

## 关键API变化

### `load_background_data`
新增 `sample_size` 参数：
- 如果指定，会在读取embedding前采样
- 如果为 `None`，返回全部数据（用于预测阶段）

```python
# 训练时：采样背景数据
background = recommender.load_background_data(
    categories=categories,
    sample_size=5000  # 只读取5000条embedding
)

# 预测时：加载全部目标数据
target = recommender.load_background_data(
    categories=categories,
    sample_size=None  # 读取全部
)
```
