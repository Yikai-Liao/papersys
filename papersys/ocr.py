import os
import json
import tempfile
import time
import pathlib
from mistralai import Mistral
from loguru import logger
from datauri import parse as parse_data_uri

def ocr_by_id(arxiv_id: str):
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": url
        },
        include_image_base64=True
    )
    return response

def ocr_by_id_batch(arxiv_ids: list[str], wait_for_completion: bool = True, poll_interval: int = 10):
    """
    Batch OCR processing for multiple arXiv papers to reduce costs.
    
    Args:
        arxiv_ids: List of arXiv IDs to process
        wait_for_completion: If True, wait for batch job to complete and return results
        poll_interval: Seconds to wait between status checks (default: 10)
    
    Returns:
        If wait_for_completion=True: dict mapping arxiv_id to OCR response
        If wait_for_completion=False: batch job object for manual tracking
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    
    # Prepare batch requests
    batch_requests = []
    for idx, arxiv_id in enumerate(arxiv_ids):
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        request = {
            "custom_id": arxiv_id,  # Use arxiv_id as custom_id for easy mapping
            "body": {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": url
                },
                "include_image_base64": True
            }
        }
        batch_requests.append(request)
    
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
        temp_file_path = f.name
    
    try:
        # Upload batch file
        with open(temp_file_path, 'rb') as f:
            batch_data = client.files.upload(
                file={
                    "file_name": "ocr_batch.jsonl",
                    "content": f
                },
                purpose="batch"
            )
        
        # Create batch job
        created_job = client.batch.jobs.create(
            input_files=[batch_data.id],
            model="mistral-ocr-latest",
            endpoint="/v1/ocr",
            metadata={"job_type": "arxiv_ocr", "num_papers": len(arxiv_ids)}
        )
        
        if not wait_for_completion:
            return created_job
        
        # Wait for completion
        logger.info(f"Batch job created: {created_job.id}")
        logger.info(f"Processing {len(arxiv_ids)} papers...")
        
        while True:
            retrieved_job = client.batch.jobs.get(job_id=created_job.id)
            status = retrieved_job.status
            
            logger.debug(f"Status: {status}")
            
            if status == "SUCCESS":
                break
            elif status in ["FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"]:
                raise Exception(f"Batch job {status}: {retrieved_job}")
            
            time.sleep(poll_interval)
        
        # Download and parse results
        output_file_stream = client.files.download(file_id=retrieved_job.output_file)
        results_content = output_file_stream.read().decode('utf-8')
        
        # Parse results and map back to arxiv_ids
        results = {}
        for line in results_content.strip().split('\n'):
            if line:
                result = json.loads(line)
                arxiv_id = result['custom_id']
                # Store the response body
                results[arxiv_id] = result.get('response', {}).get('body', result)
        
        logger.success(f"Batch processing complete! Processed {len(results)} papers.")
        return results
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def response2md(ocr_response, output_dir: str | pathlib.Path, filename: str):
    """
    Convert OCR response to markdown file with extracted images.
    
    Args:
        ocr_response: OCR response object from Mistral API or dict
        output_dir: Directory to save the markdown and images
        filename: Name of the markdown file (without extension)
    
    Returns:
        tuple: (markdown_path, image_dir, num_pages, num_images)
    """
    # 设置输出目录
    out_dir = pathlib.Path(output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 处理并保存图片
    md_parts = []
    replacements = {}
    
    # 兼容处理：支持对象和字典两种格式
    if isinstance(ocr_response, dict):
        pages = ocr_response.get("pages", [])
    else:
        pages = getattr(ocr_response, "pages", []) or []
    
    for p_idx, page in enumerate(pages):
        # 保存本页图片
        # 兼容处理：支持对象和字典两种格式
        if isinstance(page, dict):
            images = page.get("images", [])
        else:
            images = getattr(page, "images", []) or []
            
        for i_idx, img in enumerate(images):
            if isinstance(img, dict):
                b64 = img.get("image_base64")
            else:
                b64 = getattr(img, "image_base64", None)
            if not b64:
                continue
            
            try:
                # 解析 base64 数据
                data = parse_data_uri(b64)
                
                # 生成文件名
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                    "image/svg+xml": ".svg",
                }
                ext = ext_map.get(data.media_type, "")
                
                # 获取原始图片 ID（例如 "img-0.jpeg"）
                if isinstance(img, dict):
                    original_name = img.get("id", f"p{p_idx+1}-img{i_idx+1}")
                else:
                    original_name = getattr(img, "id", f"p{p_idx+1}-img{i_idx+1}")
                
                # 如果原始名称已经有扩展名，就用原始名称；否则添加扩展名
                if "." in original_name:
                    name = original_name
                else:
                    name = f"{original_name}{ext}"
                
                # 保存图片
                out_path = img_dir / name
                with open(out_path, "wb") as wf:
                    wf.write(data.data)
                
                # 记录替换规则：原始 markdown 中的图片名 -> 相对路径
                replacements[original_name] = f"images/{name}"
                
            except Exception as e:
                logger.warning(f"Failed to process image {i_idx} on page {p_idx}: {e}")
                continue
        
        # 收集 markdown 内容
        if isinstance(page, dict):
            md_parts.append(page.get("markdown", ""))
        else:
            md_parts.append(getattr(page, "markdown", ""))
    
    # 2. 合并 markdown 并替换图片路径
    full_md = "\n".join(md_parts)
    for k, v in replacements.items():
        full_md = full_md.replace(k, v)
    
    # 3. 保存 markdown 文件
    # 确保文件名有 .md 扩展名
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    
    md_path = out_dir / filename
    md_path.write_text(full_md, encoding="utf-8")
    
    num_pages = len(md_parts)
    num_images = len(replacements)
    
    logger.info(f"Markdown saved to: {md_path}")
    logger.info(f"Total pages: {num_pages} | Total images: {num_images}")

    