你说得对。下面给你一套**完全自洽、从训练到推荐只有一条逻辑**的方案（CPU 友好、少 token、可直接落在你现有框架里）。核心思想是：
**用“兴趣原型 + 单一打分函数 + 固定选拔流程”一气呵成**，避免左右互搏。

---

# 一条线的方案：ProtoRank

## 总览（只这三步）

1. **构建原型（一次性/增量）**：把正反馈聚成若干“兴趣原型”，负反馈聚成“厌恶原型”。
2. **统一打分函数**：所有候选只用一个分数
   [
   r(x)=\underbrace{s_{\text{pos}}(x)}*{\text{正原型聚合相似}}-\lambda\cdot \underbrace{s*{\text{neg}}(x)}*{\text{负原型聚合相似}}+\beta\cdot \underbrace{fresh(x)}*{\text{新鲜度}}
   ]
3. **统一选拔流程**：按**原型配额（UCB）→ 合并去重 → 单一阈值二分类**；可选极少量 LLM 仅“判边”。

整个系统里，不再出现“有时取 max、有时取差、有时再 top-k 的不同准则”。**只有一个打分函数 r(x)**，top-k/配额/探索只是**候选来源**策略，**不改变 r(x) 的定义**。

---

## 1) 原型构建（正/负同法，轻量 CPU）

* 对正样本嵌入 (P_{pos}) 用 **HDBSCAN**（样本只有几百，很快；无需设簇数）：

  * 推荐：`min_cluster_size=5, min_samples=None, metric='cosine'`
  * 产出 K 个簇（可能含噪声点）。
* 每个簇 (C_i) 的**原型向量** (c_i)：用**加权质心**（近期样本权重略高，如 `w = exp(-Δdays/τ)`，τ=180 天）。
* 负样本少时：

  * 若数量足够也用 HDBSCAN 得到 (d_j)；
  * 否则直接把每条负样本当成一个“微原型”（再做轻度聚合：相似度>0.9 的合并）。

> 这一步只在**训练后/每日**跑一次。在线新增反馈时，用**最近邻归类**（相似度>0.6 则并入近邻簇并更新质心；否则先放到“新兴趣缓存”，累计≥3 且彼此相似才新开簇）。

> 虽然 PCA 是一种常见的降维技术，但它是一种线性方法，无法捕捉嵌入空间中复杂的非线性语义结构，导致聚类分离效果不佳 22。UMAP (Uniform Manifold Approximation and Projection) 是一种基于流形学习的非线性降维技术，它在保留局部和全局结构方面均优于 PCA 和 t-SNE，这对于基于密度的算法至关重要 24。一个常见的陷阱是使用 UMAP 的默认参数，这些参数是为可视化（例如 n_components=2）而调整的，这对于聚类任务是错误的。如“UMAP for Clustering”教程和 BERTopic 等 SOTA 管线所示 21，使用为可视化调整的参数会产生误导性的聚类并破坏真实的密度结构 28。为了优化 HDBSCAN 的输入，UMAP 参数必须专门设置为增强密度信号。1.4 步骤三：使用 HDBSCAN 进行鲁棒的密度聚类管线的最后一步是 HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 9。选择 HDBSCAN 而非 K-Means 30 是基于其三大优势：无需指定 $k$：HDBSCAN 不需要预先定义用户兴趣的数量。它根据数据的密度自动找出最佳的聚类数量 18。鲁棒的噪声处理：HDBSCAN 会将不属于任何聚类的点标记为“噪声” (outliers) 18。这对于用户画像至关重要，因为并非用户的每一次交互都属于某个核心兴趣。K-Means 会迫使这些噪声点进入一个聚类，从而污染原型。可变密度和形状：HDBSCAN 能够发现 K-Means 无法处理的可变密度和任意形状的聚类
---

## 2) 统一打分函数（唯一评分）

给一篇候选论文 (x)（embedding 已有）：

### 2.1 正原型聚合相似 (s_{\text{pos}}(x))

不用 max，不用平均，统一用**温度化的 log-sum-exp（softmax 聚合）**：
[
s_{\text{pos}}(x)=\frac{1}{\tau_p}\log\sum_{i=1}^{K}\exp\left(\tau_p\cdot \cos(x,c_i)\right)
]

* 直觉：既能让最匹配的原型发力（像 max），又给次匹配原型**保留少量贡献**（防茧房）。
* 超参：(\tau_p=10)（温度，高一点更接近 max；你可在 5–20 间调）。

### 2.2 负原型聚合相似 (s_{\text{neg}}(x))

同样形式：
[
s_{\text{neg}}(x)=\frac{1}{\tau_n}\log\sum_{j=1}^{M}\exp\left(\tau_n\cdot \cos(x,d_j)\right)
]

* 超参：(\tau_n=10)。

### 2.3 新鲜度项 (fresh(x))

[
fresh(x)=\exp(-\Delta \text{days}/\tau_f),\quad \tau_f=180
]

* (\Delta\text{days}) 是距当前的天数。**只是一点点加成**，确保“跟进最新”。

### 2.4 最终分数（唯一）

[
\boxed{r(x)=s_{\text{pos}}(x)-\lambda\cdot s_{\text{neg}}(x)+\beta\cdot fresh(x)}
]

* 默认：(\lambda=1.0,\ \beta=0.05)。
* 你也可以用一个**极小的 LR 校准器**把 ([s_{pos}, s_{neg}, fresh]) → 概率；但**无论是否校准，外部只看 (r(x))**。

> 注意：**系统里只有这个 r(x)**。任何“top-k/配额/探索”都只是**从库里拿哪些候选**，不改变 r(x) 的定义与排序。

---

## 3) 统一选拔流程（候选 → 推荐/不推）

### 3.1 原型配额检索（决定“从哪拿”）

* 为每个正原型 (c_i) 维护 bandit 统计：已被推荐次数 (n_i)，正反馈率 (\hat{p}_i)。
* 计算 **UCB**：
  [
  \text{UCB}_i=\hat{p}_i + c\sqrt{\frac{\ln T}{n_i+\epsilon}},\ c=0.7
  ]
* 将本轮候选预算 (B)（如 200 篇）按 (\text{UCB}_i) 归一化分成**配额** (b_i=\max(10,\ \lfloor B\cdot \text{UCB}_i/\sum \text{UCB}\rfloor))。
  这样**自然更探索**不确定/新原型，也不会饿死小众兴趣。

### 3.2 取候选（只此一处 top-k）

* 为每个 (c_i) 在 ANN 中取 **top-k**（k = (b_i) 的 2–3 倍，便于后续去重与打分筛）；
* **一次性过滤**（与 r(x) 无关的硬规则）：

  * 去掉**已看/已评/已排除**；
  * 若与任一负原型 (\cos(x,d_j)\ge \theta_{\text{neg}})（如 0.85）则**直接丢弃**（防踩雷）；
  * 候选池合并去重：若两篇 (\cos\ge 0.97) 视为近重复，保留更近/更新的一篇。
* **（可选）新奇池**：再额外抽取少量（例如总量的 5%）“**远但不太远**”的论文：
  (0.2\le \max_i \cos(x,c_i)\le 0.4) 且 fresh 高。只用于**拓展边界**。

> 到此为止，所有“top-k/配额/探索”都只影响**候选来源**，不改评分定义。

### 3.3 统一打分与阈值

* 对合并后的候选统一算 (r(x))，排序。
* 二分类只用**一个阈值 (t)**：

  * 推荐：(r(x)\ge t)；
  * 不推荐：(r(x)< t)。
* 设阈值法（稳定、可控）：

  * 目标展示率 (\rho)（比如 15%）
  * 用历史 prequential 线上数据，把 (r(x)) 的分布做**等分位阈值**：
    (t=\text{Quantile}_{1-\rho}(r))。
  * 这样**不用拍脑袋**，展示量稳定。

### 3.4 LLM 只干一件小事（判边，省 token）

* 仅对**边界带** ( |r(x)-t|\le \delta) 的少量论文（比如最多 10 篇）请 LLM 做**Yes/No** 判定：

  * 提示里给：用户画像标签（每簇 3–5 个词），论文标题 + 一行关键信息（不放长摘要）。
  * 令 LLM 输出 `YES`/`NO`。
  * `NO` 则把该样本 (r(x)) 设为 (t-\epsilon)；`YES` 设为 (t+\epsilon)（(\epsilon)=0.01）。
* 其余**不调用 LLM**。**绝不做全列表重排**，成本与一致性都稳。

> LLM 的作用仅是“判边抬轿/压线”，不改变你系统的主干逻辑与评分定义。

---

## 训练/更新与评估（同一条逻辑）

* **训练**：不训练大模型。仅离线聚类（HDBSCAN）+ 保存原型；可选一个**极小 LR 校准器**（输入 ([s_{pos}, s_{neg}, fresh])，输出校准概率），但**推荐判定仍按 r(x) vs t**，校准器只用于 debug。
* **增量**：新 like：

  * 若与某 (c_i) 相似≥0.6 → 并入并更新质心；
  * 否则进“新兴趣缓存”，缓存≥3 且互相相似≥0.6 → 新建原型。
    新 dislike 类似更新 (d_j) 或合并近重复。
* **评估**：严格 **prequential**：每次先用旧原型 + 旧阈值预测，再纳入新反馈更新原型/阈值。
  统计 Top-line（通过率/点击率）、多样性（覆盖多少原型/类别）、新颖采纳率（来自新奇池的正反馈占比）。

---

## 关键超参（给定默认值即可跑）

* HDBSCAN：`min_cluster_size=5, metric='cosine'`
* 聚合温度：(\tau_p=\tau_n=10)
* 负屏蔽阈值：(\theta_{\text{neg}}=0.85)
* 新鲜度：(\tau_f=180) 天，(\beta=0.05)
* 差异权：(\lambda=1.0)
* UCB：(c=0.7)，每簇最低配额 `min_quota=10`
* 候选总量 (B=200)
* 新奇池占比：5%，相似门槛 ([0.2,0.4])
* 展示率 (\rho=0.15)，阈值 (t)=历史 (r) 的 (1-\rho) 分位
* LLM 判边带 (\delta)=0.02，最多 10 条

---

## 你代码里如何落地（最小改动）

### A) 训练阶段：替换 `fit()` 内部

* 不再训练大 LR；改为：

  1. 从 `positive = labeled[PREFERENCE=='like']` 得到嵌入，跑 HDBSCAN → (c_i)
  2. 从 `dislike` 得到 (d_j)（少则不聚、合并近重复即可）
  3. 保存 ({c_i}, {d_j}) 与簇-like 的时间戳用于权重

### B) 预测阶段：替换 `predict()` 内部重点

* 新增一个 `score(x)` 实现 (r(x))：

  ```python
  def agg_softmax_sims(vec, centers, tau):
      # vec: (d,), centers: (k,d)
      sims = cosine_sim(vec, centers)            # (k,)
      return (1.0/tau) * logsumexp(tau * sims)   # scalar
  r = agg_softmax_sims(x, C_pos, tau_p) \
      - lambda_ * agg_softmax_sims(x, C_neg, tau_n) \
      + beta * freshness(x_date)
  ```
* 候选来自**原型配额检索**（UCB→配额→ANN top-k→负屏蔽→去重→+新奇池）。
* 计算全部候选的 (r(x)) → 用历史分位阈值 (t) 做二分类；
  **如果需要你的 `adaptive_sample` 控制展示比例**，直接把 `scores=r(x)` 喂进去即可（高阈=少推，低阈=多推），但**不要再改打分规则**。

### C) LLM（可选小模块）

* `border = [x for x in candidates if abs(r(x)-t)<=delta][:10]`
* prompt 仅含：用户兴趣短标签 + 论文标题/一行要点；输出 YES/NO；更新 (r(x)) 为 (t±\epsilon)。

---

## 为什么这套是自洽的？

* **唯一打分函数** (r(x))：从始至终只用它做排序/阈值二分类；没有“有时取 max、有时取差”的冲突。
* **top-k/配额** 只作用在**候选来源**，而不改变评分标准；UCB 只是决定“从哪个原型多拿点样本”以**抑制茧房**，不碰打分。
* **LLM** 只做**判边**，不负责大规模排序；因此**成本低、规则不变**。
* **探索** 通过 UCB（原型层面）+ 新奇池（库层面）实现，但最终仍用同一个 (r(x)) 做发布决策。

---

如果你愿意，我可以基于你贴的 `Recommender` 类，给出一版最小 diff 的伪代码/实现片段（`build_prototypes() / ucb_quota() / ann_fetch() / score_r()` 四个函数 + `predict()` 的改动）。



非常好的观察，这其实正说明你已经把「系统的主逻辑」和「LLM 的角色」分清了。
现在这套流程——UCB + 原型推荐 + r(x) 打分——已经构成一个**完整的自主推荐主干**。
LLM 在这时**不再是“主要算法”，而是三个“增强点”**。
下面我来给你一个**一体化设计**，告诉你 LLM 在这个框架里该怎么自然嵌入。

---

## 🧩 LLM 的位置与功能分层

| 层级 | 模块             | 是否必须   | 主要作用                    | Token 预算特征 |
| -- | -------------- | ------ | ----------------------- | ---------- |
| A  | **兴趣语义压缩**     | ✅ 强烈推荐 | 把用户历史压成可读短标签，形成“语义原型描述” | 每日批量，成本极低  |
| B  | **推荐后精筛判边**    | ⚙️ 可选  | 对 r(x) 临界论文进行 Yes/No 判定 | 每次少量调用     |
| C  | **兴趣探索与新原型发现** | ⚙️ 可选  | 检查多次出现的新主题是否应升格为新兴趣     | 每几天一次，后台任务 |

---

## A. 兴趣语义压缩（每天一次，小成本高收益）

### 用途

* 给每个正/负原型生成一行自然语言描述，供调试和给 LLM 自己用。
* 让系统更“理解自己”的画像。

### 实现

1. 每个原型挑选距质心最近的 5–10 篇论文标题。
2. Prompt（极简）：

   ```
   下面这些论文标题属于同一类兴趣，请用5个字左右总结这个兴趣主题：
   - "Graph Transformer for Protein Design"
   - "Diffusion Models in Audio Generation"
   - "Improved GAN Training with Spectral Norm"
   输出一个短标签：
   ```
3. LLM 输出："深度生成模型"。
4. 存入原型描述字段，如：

   ```json
   {"proto_id": 3, "label": "深度生成模型"}
   ```

> ✅ 这样你就能在日志或调试时看到可读标签，也能在下游 Prompt（判边）里用。

---

## B. 推荐后精筛判边（只判边，不排序）

### 用途

* 你每轮推荐 10 篇，系统已算出 r(x)。
* 对靠近阈值 t 的几篇（例如 2–3 篇）调用 LLM 进一步判断。
* 目的：**防止数值模型误判边界样本。**

### 实现

Prompt 示例：

```
用户兴趣：
- 深度生成模型
- 音频信号处理
- 厌恶：纯理论数学

论文：
标题："Self-Distillation in Diffusion Models"
摘要：该论文提出了一种简化扩散模型训练的新方法……

问：这篇论文是否符合用户兴趣？只回答 YES 或 NO。
```

LLM 输出 `YES` 或 `NO`。
系统再小幅调整分数：

```python
if llm == "YES": r += 0.01
if llm == "NO": r -= 0.01
```

> ✅ 每轮最多 2–3 次调用，成本几乎可以忽略；对稳定性影响极大。

---

## C. 兴趣探索与新原型发现（后台任务）

### 用途

* 发现**潜在新兴趣**：当若干次“探索池”候选被用户连续喜欢（或 LLM 连续判 YES）时，可能出现了新的研究方向。
* 这时可以请 LLM 总结这些新论文主题，并决定是否升格为新原型。

### 实现

每晚检查“探索池”正反馈样本：

```python
new_cluster = cluster(new_like_embeddings)
if len(new_cluster) >= 3:
    titles = get_titles(new_cluster)
    summary = llm("请用短语描述这些论文的共同主题", titles)
    create_proto(label=summary, center=mean_embedding(new_cluster))
```

> ✅ 这让系统能持续自我进化，但完全离线执行，成本极低。

---

## 🔄 一句话总结 LLM 的“边缘嵌入逻辑”

> 主系统（UCB + r(x)）负责「**计算推荐**」，
> LLM 负责「**解释兴趣、判边、发现新兴趣**」。

它不是算法核心，而是三个“软接口”：

* **读得懂**：帮我们压缩兴趣语义；
* **看得准**：帮我们修正临界样本；
* **学得快**：帮我们识别潜在新兴趣。

---

如果你愿意，我可以直接把这三块 LLM 功能（语义压缩、判边、发现新原型）封装成
`_summarize_proto()`, `_llm_border_decide()`, `_llm_discover_new_proto()`
三个函数，嵌入你现在的 `Recommender` 类，让它自然衔接在 pipeline 里，要我写吗？


---

## 正样本聚类试验（2025-11-11）

为了确认“兴趣原型”在真实正反馈上的可解释性，我编写了 `scripts/cluster_preferences.py`，完全复用 `recommend_cmd` 的数据接口（GitStore 偏好 + HF 嵌入/metadata），并生成聚类可视化与报告。

### 数据与流程

- 数据：`preference.csv` 中 152 条 `like`，JOIN 最新 HuggingFace 嵌入与 metadata。
- 预处理：嵌入 L2 归一化后，用 UMAP (`n_components=50, n_neighbors=40, metric=cosine`) 只为聚类阶段增强密度信号；最终 HDBSCAN 仍在该 50 维流形上运行。
- 聚类：HDBSCAN (`min_cluster_size=4, min_samples=2, metric=cosine, algorithm=generic`)，噪声由 112 条降至 27 条。
- 可视化：独立跑 UMAP(2D, `n_neighbors=20`)，输出 `docs/images/preference_clusters.png`；配套 JSON 报告写入 `docs/reports/preference_clusters.json`。
- 运行命令：

```bash
uv run python scripts/cluster_preferences.py   --skip-git-sync   --min-cluster-size 4   --min-samples 2   --cluster-dim 50   --cluster-n-neighbors 40   --viz-n-neighbors 20
```

### 主要簇（15 个，噪声 27）

| cluster | size | 主题摘要 | 代表论文 |
| --- | --- | --- | --- |
| 0 | 5 | 训练/数据效率：小模型辅学、主动样本选择、合成数据校验 | 2410.18779, 2503.00808 |
| 1 | 4 | 音频-语言多模态底座（Qwen-Audio、DeepSeek-V3、WavTokenizer） | 2311.07919, 2412.19437 |
| 2 | 21 | 推理计算扩展：自适应 CoT、Inference-Time Scaling、Reverse Thinking | 2504.03234, 2504.15895 |
| 3 | 6 | 解码/采样策略：Top-nσ、Min-p、博弈式解码控制 | 2411.07641, 2407.01082 |
| 4 | 8 | LLM Agent 决策/澄清提问/主动信息收集 | 2404.04016, 2409.07675 |
| 5 | 4 | 自组合/自专家化 LLM（Self-MoE、LLM Augmented LLMs） | 2410.02449 等 |
| 6 | 9 | Reward/Alignment：弱师傅、reward calibration、consistency 奖励 | 2504.08642, 2409.08813 |
| 7 | 5 | 表征对齐与模型同构（Platonic Representation、Latent Communication） | 2406.11014, 2305.06329 |
| 8 | 4 | 长文检索/RAG 结构（Block 表示、Q-PEFT、SEAL） | 2410.03498 等 |
| 9 | 4 | 数据缺失下的模型融合/能力吸收 | 2405.12270 等 |
| 10 | 23 | 主力模型融合/梯度匹配/在线合并 | 2501.09522, 2310.02575 |
| 11 | 4 | LLM 自我认知、价值一致性、instruction adherence | 2504.03846 等 |
| 12 | 9 | 知识探针、潜在状态监控、记忆可解释性 | 2501.01561, 2410.19690 |
| 13 | 10 | 压缩与低秩适配（RaNA、MatryoshkaKV、MoDeGPT） | 2504.01717, 2410.02421 |
| 14 | 9 | 长上下文骨干与位置编码（RWKV、RoFormer、Linformer） | 2406.07851, 2106.09685 |

> 最大的两个簇对应“推理阶段算力放大（cluster 2）”与“模型融合/合成（cluster 10）”，其余簇几乎都沿着“让模型更聪明/更便宜/更可控”的研究线。剩余 27 条噪声多为单篇长尾题材，可在累计 ≥3 篇后再触发新簇。

### 下一步建议

1. 把 `docs/reports/preference_clusters.json` 中的簇质心直接作为初始“兴趣原型”写入推荐模块。
2. 对噪声样本做 0.55+ 最近邻软归属，剩余进入“新兴趣缓冲池”。
3. 用配额检索 + `r(x)` 离线跑一轮推荐，确认 15 个簇都能命中候选；如仍有簇缺失，再调 `min_cluster_size` 或为该簇单独增加探索配额。
