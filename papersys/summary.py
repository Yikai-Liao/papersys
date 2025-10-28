from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from google import genai
from loguru import logger




# -------------- Schema (保留 Field 描述以便作为 response_schema) --------------
class PaperSummary(BaseModel):
        # title: str = Field(description="The title of the research. For example: 'Antidistillation Sampling'.")
        # authors: List[str] = Field(description="The authors of the research. For example: ['Yash Savani', 'J. Zico Kolter'].")
        institution: List[str] = Field(
                default_factory=list,
                description=(
                        "The institution where the research was conducted. For example: "
                        "['Carnegie Mellon University', 'Stanford University', 'University of California, Berkeley']"
                ),
        )

        reasoning_step: Optional[str] = Field(
                default=None,
                description=(
                        "Just a draft for you to understand this paper and do some further reasoning here. "
                        "You need to think here, deep dive into the paper and find some interesting things, "
                        "some problems, some insights, and all the things you think that you need to think. "
                        "This is a draft, so you can write anything here, but it should be deep and help you to make the following answer better."
                ),
        )

        problem_background: Optional[str] = Field(
                default=None,
                description="The motivation, research problem, and background of this research.",
        )

        method: Optional[str] = Field(
                default=None,
                description=(
                        "The method used in this research. Its core idea, how it works, and the main steps. "
                ),
        )

        experiment: Optional[str] = Field(
                default=None,
                description=(
                        "The experiment conducted in this research. The dataset used, the experimental setup, "
                        "why it was conducted and organized like this, and the results, especially if the results match the expectation."
                ),
        )

        one_sentence_summary: Optional[str] = Field(
                default=None,
                description=(
                        "A one-sentence summary of the research. This should be a concise and clear summary "
                        "of the research, including the motivation, method, and results."
                ),
        )

        slug: Optional[str] = Field(
                default=None,
                description=(
                        "A URL-friendly string that summarizes the title of the research, such as 'antidistillation-sampling'."
                ),
        )

        keywords: List[str] = Field(
                default_factory=list,
                description=(
                        "When extracting keywords, each word should be capitalized. Spaces can be used within keywords, "
                        "such as 'Proxy Model'. Keywords are used to discover connections within the article, so please use more general keywords. "
                        "Do not add more than 6 keywords for 1 paper."
                ),
        )

        further_thoughts: Optional[str] = Field(
                default=None,
                description=(
                        "Any kind of further thoughts, but it should be deep and insightful. It could be diverse, and related to other areas or articles, "
                        "but you need to find the relation and make it insightful."
                ),
        )


# -------------- Prompt 模板（保留 notebook 中的提示内容与 keyword 列表） --------------
LANG = "中文"

# 这是一个综合关键词列表（示例），summarize 时会建议模型优先使用这些概念级别的关键词
KEYWORDS_JSON = r"""
{
        "Learning Paradigms": [
            "Supervised Learning",
            "Unsupervised Learning",
            "Self-Supervised Learning",
            "Reinforcement Learning",
            "Transfer Learning",
            "Few-Shot Learning",
            "Zero-Shot Learning",
            "Online Learning",
            "Active Learning",
            "Continual Learning",
            "Federated Learning",
            "Meta-Learning",
            "Imitation Learning",
            "Contrastive Learning"
        ],
        "Model Architectures": [
            "Transformer",
            "CNN",
            "RNN",
            "GNN",
            "MLP",
            "Autoencoder",
            "State Space Model"
        ],
        "Fundamental Tasks & Capabilities": [
            "Classification",
            "Regression",
            "Detection",
            "Segmentation",
            "Prediction",
            "Reasoning",
            "Planning",
            "Control",
            "Translation",
            "Representation Learning",
            "Embeddings"
        ],
        "Data Concepts & Handling": [
            "Dataset",
            "Benchmark",
            "Data Augmentation",
            "Preprocessing",
            "Feature Engineering",
            "Unstructured Data",
            "Tabular Data",
            "Time Series Data",
            "Graph Data",
            "Multimodal Data",
            "Synthetic Data",
            "Tokenization"
        ],
        "Large Models & Foundation Models": [
        "Large Language Model",
        "Vision Foundation Model",
            "Foundation Model",
            "Pre-training",
            "Fine-tuning",
            "Instruction Tuning",
            "Parameter-Efficient Fine-Tuning",
            "Low-Rank Adaptation",
            "Prompt Engineering",
            "In-Context Learning",
            "Emergent Abilities",
            "Scaling Laws",
            "Long Context"
        ],
        "Generative AI": [
            "Generative AI",
            "Generative Modeling",
            "Diffusion Model",
            "Generative Adversarial Network",
            "Flow Matching",
            "Normalizing Flow",
            "Image Generation",
            "Video Generation",
            "Audio Generation",
            "Text-to-Image",
            "Text-to-Video",
            "Molecule Generation",
            "Code Generation"
        ],
        "Trust, Ethics, Safety & Alignment": [
            "Alignment",
            "DPO",
            "RLHF",
            "Safety",
            "Fairness",
            "Interpretability",
            "Robustness",
            "AI Ethics",
            "Responsible AI",
            "Trustworthy AI",
            "Privacy-Preserving Machine Learning"
        ],
        "System Properties & Interaction": [
            "Efficiency",
            "Test Time",
            "Adaptive Systems",
            "Multimodality",
            "Multimodal Systems",
            "Human-AI Interaction"
        ],
        "AI Application Domains & Cross-cutting Fields": [
            "Robotics",
            "Agent",
            "Multi-Agent",
            "RAG",
            "Recommender Systems",
            "AI for Drug Discovery",
            "AI for Science",
            "AI in Finance",
            "AI in Security"
        ]
    }
"""


PROMPT_TEMPLATE = (
        "You are now a top research expert. Please carefully read the specified paper, make sure to fully understand the core ideas of the paper, and then explain it accurately and in detail. "
        "Treat this summarization task as a peer review: be careful, serious, and critical — but do not criticize for criticism's sake; focus on core idea, methods and experiments.\n\n"
        "Answer these questions and return a JSON matching the PaperSummary schema: \n"
        "1) institution: participating institutions;\n"
        "2) problem_background: starting point and key problems solved;\n"
        "3) method: the method and core idea;\n"
        "4) experiment: experimental setup and whether results support claims;\n"
        "5) inspired_idea / further_thoughts;\n"
        "6) one_sentence_summary: concise one-sentence summary;\n"
        "7) keywords: up to 6 concept-level keywords (use given keyword list first).\n\n"
        "Use the following keyword ontology to prefer concept-level keywords: \n"
        f"{KEYWORDS_JSON}\n\n"
        f"Language for answer: {LANG}.\n"
        "Return only JSON that can be parsed into the PaperSummary schema. Escape newlines inside strings properly."
)


# -------------- Helper functions --------------

def _make_client(api_key: Optional[str] = None) -> genai.Client:
    """创建 genai.Client，如果未提供 api_key 则从环境变量读取（需要先 load_dotenv）"""
    import os
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in environment or .env file, or pass api_key parameter."
        )
    return genai.Client(api_key=api_key)


def summarize_texts(
    inputs: Dict[str, str],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    use_batch: bool = True,
    max_wait_time: int = 600,
    poll_interval: int = 3,
) -> Dict[str, PaperSummary]:
    """
    对 dict[arxiv_id -> paper_text] 批量调用 Gemini 进行摘要，返回 dict[arxiv_id -> PaperSummary]。
    
    优先尝试使用 Batch API（类似 embedding.py 的 batches.create 方式），如果失败则回退到逐文档调用。
    
    Args:
        inputs: 映射 arxiv id -> 论文全文（markdown 或纯文本）
        model: Gemini 模型名称，默认 "gemini-2.5-flash"
        api_key: Gemini API key（可选，未提供则从环境变量读取）
        use_batch: 是否尝试使用 Batch API（默认 True）
        max_wait_time: Batch job 最大等待时间（秒）
        poll_interval: 轮询间隔（秒）
        
    Returns:
        dict[arxiv_id, PaperSummary]: 摘要结果
    """
    client = _make_client(api_key=api_key)
    ids = list(inputs.keys())
    texts = [inputs[k] for k in ids]
    
    results: Dict[str, PaperSummary] = {}
    
    # 尝试 Batch API（参考 embedding.py 的 create_embeddings 方式）
    if use_batch and len(texts) > 0:
        try:
            logger.info(f"尝试使用 Batch API 对 {len(texts)} 篇论文进行摘要...")
            
            # 构造 inlined_requests（每个文档一个 generate request）
            # 参考 embedding 中的 inlined_requests 格式，但这里用 generate_content 的 request
            # 注意：Batch API 对 generate_content 的支持可能因 SDK 版本有差异，这里 best-effort
            requests = []
            for text in texts:
                req_content = f"{PROMPT_TEMPLATE}\n\n===== PAPER CONTENT =====\n\n{text}"
                requests.append({"contents": req_content})
            
            # 尝试调用 batches.create（类似 embedding 的方式）
            # 注意：generate_content 的 batch 接口名可能是 batches.create 而不是 create_embeddings
            batch_job = client.batches.create(
                model=model,
                src={"inlined_requests": requests},
                config={"display_name": f"papersys-summary-batch-{len(texts)}"},
            )
            
            job_name = batch_job.name
            logger.info(f"Batch job 已创建: {job_name}，开始轮询...")
            
            completed_states = {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }
            
            import time
            start = time.time()
            while True:
                batch_job = client.batches.get(name=job_name)
                state = getattr(batch_job.state, "name", batch_job.state)
                if isinstance(state, str):
                    state_str = state
                else:
                    state_str = str(state)
                
                if state_str in completed_states:
                    break
                    
                elapsed = time.time() - start
                if elapsed > max_wait_time:
                    raise TimeoutError(f"Batch job 超时（{max_wait_time}秒）")
                
                logger.debug(f"Batch job 状态: {state_str}, 已等待 {elapsed:.1f}秒")
                time.sleep(poll_interval)
            
            if state_str != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch job 失败，状态: {state_str}")
            
            logger.success("Batch job 成功完成，开始解析结果...")
            
            # 尝试提取 inlined_responses（类似 embedding 的 inlined_embed_content_responses）
            dest = getattr(batch_job, "dest", None)
            if dest is None:
                raise ValueError("batch_job.dest 为空")
            
            inlined = getattr(dest, "inlined_responses", None)
            if inlined is None:
                raise ValueError("dest.inlined_responses 为空")
            
            if len(inlined) != len(ids):
                logger.warning(f"返回的 response 数量 ({len(inlined)}) 与输入不匹配 ({len(ids)})")
            
            # 解析每个 response
            import json
            for i, (aid, resp) in enumerate(zip(ids, inlined)):
                try:
                    # resp 可能有 .response 属性，里面包含生成的内容
                    # 尝试提取文本（可能在 .response.text 或 .response.candidates[0].content.parts[0].text）
                    raw_text = None
                    if hasattr(resp, "response"):
                        response_obj = resp.response
                        # 尝试 .text
                        if hasattr(response_obj, "text"):
                            raw_text = response_obj.text
                        # 尝试 .candidates
                        elif hasattr(response_obj, "candidates") and response_obj.candidates:
                            cand = response_obj.candidates[0]
                            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                                parts = cand.content.parts
                                if parts and hasattr(parts[0], "text"):
                                    raw_text = parts[0].text
                    
                    if raw_text is None:
                        logger.warning(f"无法从 batch response[{i}] 提取文本，跳过 {aid}")
                        results[aid] = PaperSummary()
                        continue
                    
                    # 尝试解析为 JSON
                    try:
                        parsed = json.loads(raw_text)
                        results[aid] = PaperSummary(**parsed)
                    except json.JSONDecodeError:
                        # 如果不是 JSON，尝试提取 JSON block（```json ... ```）
                        import re
                        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                        if match:
                            parsed = json.loads(match.group(1))
                            results[aid] = PaperSummary(**parsed)
                        else:
                            # 无法解析，放入 reasoning_step
                            results[aid] = PaperSummary(reasoning_step=raw_text[:1000])
                    
                except Exception as e:
                    logger.error(f"解析 batch response[{i}] 失败: {e}, aid={aid}")
                    results[aid] = PaperSummary()
            
            logger.success(f"Batch API 成功处理 {len(results)} 篇论文")
            return results
            
        except Exception as e:
            logger.warning(f"Batch API 失败: {e}，回退到逐文档调用")
    
    # 回退：逐文档调用 models.generate_content
    logger.info(f"使用逐文档调用模式处理 {len(texts)} 篇论文...")
    import json
    for aid, text in inputs.items():
        try:
            prompt = f"{PROMPT_TEMPLATE}\n\n===== PAPER CONTENT =====\n\n{text}"
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PaperSummary,
                },
            )
            
            # 提取文本
            raw_text = None
            if hasattr(response, "text"):
                raw_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    parts = cand.content.parts
                    if parts and hasattr(parts[0], "text"):
                        raw_text = parts[0].text
            
            if raw_text is None:
                logger.warning(f"无法提取 response text for {aid}")
                results[aid] = PaperSummary()
                continue
            
            # 解析 JSON
            try:
                parsed = json.loads(raw_text)
                results[aid] = PaperSummary(**parsed)
            except json.JSONDecodeError:
                # 尝试提取 JSON block
                import re
                match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    results[aid] = PaperSummary(**parsed)
                else:
                    results[aid] = PaperSummary(reasoning_step=raw_text[:1000])
                    
        except Exception as e:
            logger.error(f"处理 {aid} 时出错: {e}")
            results[aid] = PaperSummary()
    
    logger.success(f"逐文档调用完成，共处理 {len(results)} 篇论文")
    return results


def summarize_from_path_map(
    path_map: Dict[str, Path],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    use_batch: bool = True,
    max_wait_time: int = 600,
    poll_interval: int = 3,
) -> Dict[str, PaperSummary]:
    """
    从路径映射读取 markdown 文件并进行摘要。
    
    Args:
        path_map: dict[arxiv_id -> Path]，Path 可以是：
            - 包含 <arxiv_id>.md 文件的目录（如 data/ocr_responses_example/2510.20766）
            - 直接指向 .md 文件的路径
        model: Gemini 模型名称
        api_key: API key（可选）
        use_batch: 是否使用 Batch API
        max_wait_time: Batch job 最大等待时间
        poll_interval: 轮询间隔
        
    Returns:
        dict[arxiv_id, PaperSummary]: 摘要结果
    """
    inputs: Dict[str, str] = {}
    
    for aid, p in path_map.items():
        p = Path(p)
        
        # 如果是目录，查找 <aid>.md
        if p.is_dir():
            md_file = p / f"{aid}.md"
        else:
            # 如果是文件，直接使用
            md_file = p if p.suffix.lower() == ".md" else p / f"{aid}.md"
        
        if not md_file.exists():
            logger.warning(f"文件不存在: {md_file}, 跳过 {aid}")
            continue
        
        try:
            text = md_file.read_text(encoding="utf-8")
            inputs[aid] = text
        except Exception as e:
            logger.error(f"读取 {md_file} 失败: {e}, 跳过 {aid}")
            continue
    
    if not inputs:
        logger.warning("没有读取到任何有效的 markdown 文件")
        return {}
    
    logger.info(f"成功读取 {len(inputs)} 篇论文，开始调用摘要...")
    return summarize_texts(
        inputs=inputs,
        model=model,
        api_key=api_key,
        use_batch=use_batch,
        max_wait_time=max_wait_time,
        poll_interval=poll_interval,
    )
