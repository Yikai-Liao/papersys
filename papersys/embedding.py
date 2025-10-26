import os
import time
from typing import Optional
from collections import deque

import numpy as np
import polars as pl 
from google import genai
from google.genai import types
from loguru import logger
from .database.name import *

def collect_content(df: pl.DataFrame) -> list[str]:
    CONTENT = "content"
    df = df.lazy().with_columns(
        pl.col(TITLE).fill_null(""),
        pl.col(ABSTRACT).fill_null("")
    )
    contents = (
        df.select(
            (pl.col(TITLE) + "\n" + pl.col(ABSTRACT)).alias(CONTENT)
        )
        .collect()
        .to_series()
        .to_list()
    )
    return contents
    

class RateLimiter:
    """简单的速率限制器，基于滑动窗口算法"""
    
    def __init__(self, tokens_per_minute: int = 1_000_000, window_seconds: int = 60):
        """
        Args:
            tokens_per_minute: 每分钟允许的最大 token 数
            window_seconds: 时间窗口（秒）
        """
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self.requests = deque()  # (timestamp, token_count)
        
    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量（粗略估计：1 token ≈ 4 字符）"""
        return len(text) // 4 + 1
    
    def estimate_batch_tokens(self, texts: list[str]) -> int:
        """估算批次的总 token 数"""
        return sum(self.estimate_tokens(text) for text in texts)
    
    def wait_if_needed(self, token_count: int):
        """如果需要，等待直到可以发送请求"""
        now = time.time()
        
        # 移除窗口外的旧请求
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0][0] < cutoff:
            self.requests.popleft()
        
        # 计算当前窗口内的 token 总数
        current_tokens = sum(count for _, count in self.requests)
        
        # 如果加上新请求会超过限制，等待
        if current_tokens + token_count > self.tokens_per_minute:
            # 计算需要等待的时间（等到最早的请求过期）
            if self.requests:
                oldest_time = self.requests[0][0]
                wait_time = oldest_time + self.window_seconds - now + 1  # +1 秒缓冲
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit approaching. Current tokens: {current_tokens}, "
                        f"Requesting: {token_count}. Waiting {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                    # 递归调用，重新检查
                    return self.wait_if_needed(token_count)
        
        # 记录这次请求
        self.requests.append((now, token_count))


def google_batch_embedding_with_rate_limit(
    model: str,
    inputs: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None,
    max_wait_time: int = 600,
    poll_interval: int = 5,
    dtype = np.float32,
    batch_size: int = 50000,  # 每批次最大文档数
    tokens_per_minute: int = 900_000,  # 保守估计，留 10% 余量
    max_retries: int = 3,
) -> np.ndarray:
    """
    使用 Google Gemini Batch Embedding API 获取文本嵌入向量，带速率限制
    
    Args:
        model: 模型名称，如 "gemini-embedding-001"
        inputs: 要嵌入的文本列表
        task_type: 任务类型
        output_dimensionality: 输出维度
        api_key: Google API Key
        max_wait_time: 最大等待时间（秒）
        poll_interval: 轮询间隔（秒）
        dtype: 输出数据类型
        batch_size: 每批次最大文档数，避免单次请求过大
        tokens_per_minute: 每分钟允许的最大 token 数
        max_retries: 最大重试次数
        
    Returns:
        np.ndarray: 形状为 (len(inputs), embedding_dim) 的嵌入矩阵
    """
    total_inputs = len(inputs)
    
    # 如果输入较少，直接调用原函数
    if total_inputs <= batch_size:
        logger.info(f"Processing {total_inputs} documents in a single batch")
        return google_batch_embedding(
            model=model,
            inputs=inputs,
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            api_key=api_key,
            max_wait_time=max_wait_time,
            poll_interval=poll_interval,
            dtype=dtype,
        )
    
    # 分批处理
    logger.info(f"Processing {total_inputs} documents in batches of {batch_size}")
    rate_limiter = RateLimiter(tokens_per_minute=tokens_per_minute)
    
    all_embeddings = []
    num_batches = (total_inputs + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_inputs)
        batch_inputs = inputs[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} "
                   f"(documents {start_idx} to {end_idx - 1})")
        
        # 估算 token 数并等待（如果需要）
        estimated_tokens = rate_limiter.estimate_batch_tokens(batch_inputs)
        logger.debug(f"Estimated tokens for this batch: {estimated_tokens}")
        rate_limiter.wait_if_needed(estimated_tokens)
        
        # 重试逻辑
        for attempt in range(max_retries):
            try:
                batch_embeddings = google_batch_embedding(
                    model=model,
                    inputs=batch_inputs,
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                    api_key=api_key,
                    max_wait_time=max_wait_time,
                    poll_interval=poll_interval,
                    dtype=dtype,
                )
                all_embeddings.append(batch_embeddings)
                logger.success(f"Batch {batch_idx + 1}/{num_batches} completed successfully")
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30  # 30, 60, 90 秒
                        logger.warning(
                            f"Rate limit error on batch {batch_idx + 1}, "
                            f"attempt {attempt + 1}/{max_retries}. "
                            f"Waiting {wait_time} seconds before retry..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed batch {batch_idx + 1} after {max_retries} attempts")
                        raise
                else:
                    logger.error(f"Unexpected error on batch {batch_idx + 1}: {error_msg}")
                    raise
    
    # 合并所有批次的嵌入
    return np.vstack(all_embeddings)


def google_batch_embedding(
    model: str,
    inputs: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None,
    max_wait_time: int = 600,
    poll_interval: int = 5,
    dtype = np.float32,
) -> np.ndarray:
    """
    使用 Google Gemini Batch Embedding API 获取文本嵌入向量（价格为正常 API 的 50%）
    
    Args:
        model: 模型名称，如 "gemini-embedding-001"
        inputs: 要嵌入的文本列表
        task_type: 任务类型，可选值:
            - "RETRIEVAL_DOCUMENT": 文档检索（默认）
            - "RETRIEVAL_QUERY": 查询检索
            - "SEMANTIC_SIMILARITY": 语义相似度
            - "CLASSIFICATION": 分类
            - "CLUSTERING": 聚类
        output_dimensionality: 输出维度，可选 128-3072，推荐 768, 1536, 3072
        api_key: Google API Key，如果不提供则从环境变量 GEMINI_API_KEY 读取
        max_wait_time: 最大等待时间（秒），默认 600 秒
        poll_interval: 轮询间隔（秒），默认 5 秒
        
    Returns:
        np.ndarray: 形状为 (len(inputs), embedding_dim) 的嵌入矩阵
        
    Raises:
        ValueError: 如果 API key 未提供或 job 失败
        TimeoutError: 如果超过最大等待时间
    """
    # 获取 API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
    
    # 创建客户端
    client = genai.Client(api_key=api_key)
    
    # 准备配置
    embed_config = types.EmbedContentConfig(task_type=task_type)
    if output_dimensionality is not None:
        embed_config.output_dimensionality = output_dimensionality
    
    # 创建 batch embedding job
    batch_job = client.batches.create_embeddings(
        model=model,
        src={
            'inlined_requests': types.EmbedContentBatch(
                contents=inputs,
                config=embed_config
            )
        },
        config={'display_name': f"batch-embedding-{len(inputs)}-texts"},
    )
    
    # 轮询等待 job 完成
    job_name = batch_job.name
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    
    start_time = time.time()
    while True:
        batch_job = client.batches.get(name=job_name)
        current_state = batch_job.state.name
        
        if current_state in completed_states:
            break
        
        # 检查超时
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_time:
            raise TimeoutError(f"Batch job did not complete within {max_wait_time} seconds")
        
        time.sleep(poll_interval)
    
    # 检查 job 状态
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        error_msg = f"Batch job failed with state: {batch_job.state.name}"
        if hasattr(batch_job, 'error') and batch_job.error:
            error_msg += f"\nError: {batch_job.error}"
        raise ValueError(error_msg)
    
    # 提取嵌入向量
    if not (batch_job.dest and batch_job.dest.inlined_embed_content_responses):
        raise ValueError("No embedding responses found in batch job result")
    
    responses = batch_job.dest.inlined_embed_content_responses
    embeddings = [resp.response.embedding.values for resp in responses]
    
    return np.array(embeddings, dtype=dtype)
    
    