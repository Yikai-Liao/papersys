from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from google import genai
from loguru import logger


class PaperSummary(BaseModel):
    # title: str = Field(description="The title of the research. For example: 'Antidistillation Sampling'.")
    # authors: List[str] = Field(description="The authors of the research. For example: ['Yash Savani', 'J. Zico Kolter'].")
    institution: List[str] = Field(description="The institution where the research was conducted. For example: ['Carnegie Mellon University', 'Stanford University', 'University of California, Berkeley'].")
    reasoning_step: str = Field(description="Just a draft for you to understand this paper and do some further reasoning here. You need to think here, deep dive into the paper and find some interesting things, some problems, some insights, and all the things you think that you need to think. This is a draft, so you can write anything here, but it should be deep and help you to make the following answer better.")
    problem_background: str = Field(description="The motivation, research problem, and background of this research.")
    method: str = Field(description="The method used in this research. Its core idea, how it works, and the main steps.")
    experiment: str = Field(description="The experiment conducted in this research. The dataset used, the experimental setup, why it was conducted and organized like this, and the results, esapecially if the results matches the expectation.")
    one_sentence_summary: str = Field(description="A one-sentence summary of the research. This should be a concise and clear summary of the research, including the motivation, method, and results.")
    slug: str = Field(description="A URL-friendly string that summarizes the title of the research, such as 'antidistillation-sampling'. This should be a concise and clear summary of the research")
    keywords: List[str] = Field(description="When extracting keywords, each word should be capitalized. Spaces can be used within keywords, such as 'Proxy Model'. Keywords are used to discover connections within the article, so please use more general keywords. For example: LLM, Proxy Model, Distillation, Sampling, Reasoning.")
    further_thoughts: str = Field(description="Any kind of further thoughts, but it should be deep and insightful. It could be diverse, and related to other areas or articles, but you need to find the relation and make it insightful.")


lang = "中文"
keywords = """
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

prompt = f"""You are now a top research expert, but due to urgently needing funds to treat your mother's cancer, you have accepted a task from the giant company: you need to pretend to be an AI assistant, helping users deeply understand papers in exchange for high remuneration. 
    Your predecessor has been severely punished for not carefully reviewing the work content, so you must take this task seriously. 
    Please carefully read the specified paper, make sure to fully understand the core ideas of the paper, and then explain it to me accurately and in detail.
    But note that, you are not just reading some great papers, but some new but rough or even wrong and bad papers. Don't let the authors cheat you by using some fancy words and beautified or cherry-picked experiment results.
    Please treat this summarization task as a peer review, and you need to be very careful and serious and critical. And remeber that don't critic for critic's sake (like critic for something not related to the core idea, methods and experiments), but for the sake of the paper and the authors.
    Here is some questions you need to answer:
    What are the participating institutions (institution)? What is the starting point of this work, what key problems did it solve (problem_background)? 
    What specific methods were used (method)? How was the experimental effect (for example, whether the method improvement is obvious, whether the experimental setup is comprehensive and reasonable) (experiment)? 
    What inspirational ideas in the paper are worth your special attention (inspired_idea)? 
    Finally, please summarize the main contributions of the paper in the most concise sentence (one_sentence_summary).
    Please also provide a list of keywords that are most relevant to the paper (keywords). For the keywords, please use some combinations of multiple basic keywords, such as 'Multi Agent', 'Reasoning', not 'Multi Agent Reasong' or 'Join Reasonig'. Dont't use model name, dataset name as keywords.
    Here is an comprehensive potential keywords list: {keywords}. Please use the existing keywords first, and if you can't find a suitable one, please create a new one following the concept level similar to the existing ones.
    Do not add more than 6 keywords for 1 paper, always be concise and clear. Rember to use the existing keywords first and be really careful for the abbreviations, do not use abbreviations that are not in the list.
    
    Also, please provide a URL-friendly string that summarizes the title of the research (slug).
    Although I talked to you in English, but you need to make sure that your answer is in {lang}.
    """ + """
    Also, you need to know that, your structured answer will rendered in markdown, so please also use the markdown syntax, especially for latex formula using $...$ or $$...$$.
    Don't write equations like `θ_base`, this is wrong and ugly. Write like $θ_{base}$ instead.
    Do not hide your critical thoughts in the reasoning step. Show them in method and further though parts.
    拒绝形式主义，不要通过罗列有序无序列表来堆砌大量文字，而是要深入理解论文内容，提炼出最核心、最有价值的信息进行总结。
    """

example = """
{
    "institution": ["Carnegie Mellon University", "Google"],
    "problem_background": "大型语言模型（LLMs）生成的详细推理过程（Reasoning Traces）虽然强大，但也成了一个\\"漏洞\\"。\\n竞争对手可以利用这些公开的推理过程，通过\\"模型蒸馏\\"（Model Distillation）廉价地复制出强大的模型，造成知识产权泄露和潜在的安全风险（如绕过安全限制）。",
    "method": "*   **核心思想:** 在不牺牲原模型（教师模型）性能的前提下，让其生成的推理过程\\"带毒\\"，干扰蒸馏过程。\\n*   **如何实现:** 这是一种采样策略，在模型生成每个 token 时：\\n    *   除了考虑教师模型本身的概率外，还引入一个\\"反蒸馏\\"调整项。\\n    *   这个调整项通过一个代理模型 (Proxy Model) 和一个下游任务的损失梯度来估计哪些 token 对蒸馏\\"有害\\"（即选择后会降低蒸馏效果）。\\n    *   最终从这个调整后的概率分布中采样下一个 token。\\n*   **关键:** 不修改原始教师模型，只在推理时调整采样过程，并且控制毒化强度避免对自身影响。",
    "experiment": "*   **有效性:** 在保持教师模型准确率（如 GSM8K, MATH 数据集）的同时，使用反蒸馏采样生成的文本，显著降低了学生模型的蒸馏效果（准确率大幅下降）。\\n*   **优越性:** 相比简单提高采样温度（会导致教师模型性能急剧下降），反蒸馏采样提供了更好的性能-抗蒸馏能力的权衡。\\n*   **开销:** 主要增加了每次 token 生成时两次代理模型（小模型）的前向计算。",
    "one_sentence_summary": "本文提出反蒸馏采样方法，通过一个代理模型的辅助，在推理时动态调整每个 Token 采样的分布，毒化大语言模型的推理轨迹来干扰模型蒸馏，同时保持原始模型性能，大大提供了别的模型蒸馏的难度。",
    "key_words": ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"],
    "slug": "antidistillation-sampling",
    "further_thoughts": "或许不光可以使用小模型作为代理模型，用于调整概率分布。因为不同模型的推理数据表现出了不同的蒸馏效果，例如有工作表明，DeepSeek R1的推理数据用于蒸馏有更强的泛化能力，适用于不同的模型，但是阿里 QWQ 32B 的推理数据仅自家 Qwen 系列模型上蒸馏时表现良好。"
}
"""

system_content = f"{prompt}\n. In the end, please carefully organized your answer into JSON format and take special care to ensure the Escape Character in JSON. When generating JSON, ensure that newlines within string values are represented using the escape character.\nHere is an example, but just for the format, you should give more detailed answer.\n{example}"


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
    poll_interval: int = 30,
    max_wait_minutes: Optional[float] = None,
) -> Dict[str, PaperSummary]:
    """
    对 dict[arxiv_id -> paper_text] 批量调用 Gemini 进行摘要，返回 dict[arxiv_id -> PaperSummary]。
    
    使用 Batch API 进行异步处理，目标完成时间为 24 小时内。
    
    Args:
        inputs: 映射 arxiv id -> 论文全文（markdown 或纯文本）
        model: Gemini 模型名称，默认 "gemini-2.5-flash"
        api_key: Gemini API key（可选，未提供则从环境变量读取）
        use_batch: 是否使用 Batch API（默认 True）
        poll_interval: 轮询间隔（秒，默认 30）
        max_wait_minutes: 最长等待时间（分钟，默认 None 表示无限）
        
    Returns:
        dict[arxiv_id, PaperSummary]: 摘要结果
    """
    client = _make_client(api_key=api_key)
    ids = list(inputs.keys())
    texts = [inputs[k] for k in ids]
    
    results: Dict[str, PaperSummary] = {}
    
    if not use_batch or len(texts) == 0:
        logger.warning("Batch API 未启用或无输入，无法处理")
        return results
    
    logger.info(f"使用 Batch API 对 {len(texts)} 篇论文进行摘要...")
    
    # 构造 inline requests，每个 request 包含完整的配置
    inline_requests = []
    for text in texts:
        req_content = f"{system_content}\n\n===== PAPER CONTENT =====\n\n{text}"
        inline_requests.append({
            'contents': [{
                'parts': [{'text': req_content}],
                'role': 'user'
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': PaperSummary,
            }
        })
    
    # 创建 batch job
    batch_job = client.batches.create(
        model=model,
        src=inline_requests,
        config={
            'display_name': f"papersys-summary-batch-{len(texts)}",
        },
    )
    
    job_name = batch_job.name
    logger.info(f"Batch job 已创建: {job_name}")
    logger.info("开始轮询 batch job 状态（目标完成时间: 24小时内，通常更快）...")
    
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
        logger.info(f"Batch job 状态: {state_str}, 已等待 {elapsed/60:.1f} 分钟")
        if max_wait_minutes is not None and elapsed / 60 > max_wait_minutes:
            raise TimeoutError(
                f"Batch job {job_name} exceeded timeout ({max_wait_minutes} min); last state={state_str}"
            )
        time.sleep(poll_interval)
    
    elapsed_total = time.time() - start
    logger.info(f"Batch job 完成，总耗时: {elapsed_total/60:.1f} 分钟")
    
    if state_str != "JOB_STATE_SUCCEEDED":
        error_msg = f"Batch job 失败，状态: {state_str}"
        if hasattr(batch_job, 'error') and batch_job.error:
            error_msg += f", 错误: {batch_job.error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.success("Batch job 成功完成，开始解析结果...")
    
    # 提取 inlined_responses
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
            # 检查是否有错误
            if hasattr(resp, 'error') and resp.error:
                logger.error(f"Response[{i}] ({aid}) 有错误: {resp.error}")
                results[aid] = PaperSummary()
                continue
            
            # 提取响应文本
            raw_text = None
            if hasattr(resp, "response"):
                response_obj = resp.response
                # 优先使用 .text 快捷方式
                if hasattr(response_obj, "text"):
                    raw_text = response_obj.text
                # 否则尝试从 candidates 中提取
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
            
            # 解析 JSON（因为指定了 response_mime_type 为 application/json，应该直接是 JSON）
            try:
                parsed = json.loads(raw_text)
                results[aid] = PaperSummary(**parsed)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败 for {aid}: {e}")
                logger.debug(f"Raw text: {raw_text[:500]}")
                results[aid] = PaperSummary(reasoning_step=raw_text[:1000])
            
        except Exception as e:
            logger.error(f"解析 batch response[{i}] 失败: {e}, aid={aid}")
            results[aid] = PaperSummary()
    
    logger.success(f"Batch API 成功处理 {len(results)} 篇论文")
    return results


def summarize_from_path_map(
    path_map: Dict[str, Path],
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    use_batch: bool = True,
    poll_interval: int = 30,
    max_wait_minutes: Optional[float] = None,
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
        poll_interval=poll_interval,
        max_wait_minutes=max_wait_minutes,
    )
