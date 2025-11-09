import os
import json
import tempfile
import time
import pathlib
import requests
from mistralai import Mistral
from loguru import logger
from datauri import parse as parse_data_uri

# arXiv rate limiting: 3 seconds between requests to avoid 403/captcha
ARXIV_DOWNLOAD_DELAY = 3.0

def download_arxiv_pdf(arxiv_id: str, output_path: pathlib.Path, retry_delay: float = ARXIV_DOWNLOAD_DELAY, max_retries: int = 3) -> bool:
    """
    Download PDF from arXiv with rate limiting to avoid 403/captcha.
    
    Args:
        arxiv_id: arXiv paper ID (e.g., "2201.04234")
        output_path: Path to save the PDF file
        retry_delay: Seconds to wait between requests (default: 3.0)
        max_retries: Maximum number of retry attempts
    
    Returns:
        True if download successful, False otherwise
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {
        "User-Agent": "PaperSys/0.1 (Educational Research Tool; mailto:research@example.com)"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            # Rate limiting: wait before request
            if attempt > 1:
                time.sleep(retry_delay)
            
            logger.debug(f"Downloading {arxiv_id} from {url} (attempt {attempt}/{max_retries})")
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded {arxiv_id} successfully")
                return True
            elif response.status_code == 403:
                logger.warning(f"Got 403 for {arxiv_id}, waiting longer before retry...")
                time.sleep(retry_delay * 2)  # Wait longer on 403
            else:
                logger.warning(f"Download failed for {arxiv_id}: HTTP {response.status_code}")
                
        except requests.RequestException as exc:
            logger.warning(f"Download error for {arxiv_id} (attempt {attempt}/{max_retries}): {exc}")
            
        # Wait before next retry (except after last attempt)
        if attempt < max_retries:
            time.sleep(retry_delay)
    
    logger.error(f"Failed to download {arxiv_id} after {max_retries} attempts")
    return False

def ocr_by_id(arxiv_id: str, pdf_cache_dir: pathlib.Path | None = None, cleanup_pdf: bool = False):
    """
    OCR a single arXiv paper by downloading PDF first, then uploading to Mistral.
    
    Args:
        arxiv_id: arXiv paper ID
        pdf_cache_dir: Directory to cache downloaded PDFs (default: temp dir)
        cleanup_pdf: Whether to delete the PDF after processing
    
    Returns:
        OCR response object from Mistral API
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    
    # Prepare PDF cache directory
    if pdf_cache_dir is None:
        pdf_cache_dir = pathlib.Path(tempfile.gettempdir()) / "papersys_ocr_cache"
    pdf_path = pdf_cache_dir / f"{arxiv_id}.pdf"
    
    # Download PDF from arXiv
    if not pdf_path.exists():
        success = download_arxiv_pdf(arxiv_id, pdf_path)
        if not success:
            raise RuntimeError(f"Failed to download PDF for {arxiv_id}")
    else:
        logger.debug(f"Using cached PDF for {arxiv_id}")
    
    try:
        # Upload to Mistral Cloud
        with open(pdf_path, 'rb') as f:
            uploaded_file = client.files.upload(
                file={
                    "file_name": f"{arxiv_id}.pdf",
                    "content": f,
                },
                purpose="ocr"
            )
        
        # Get signed URL
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        
        # Process OCR
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True
        )
        
        # Clean up Mistral Cloud file
        client.files.delete(file_id=uploaded_file.id)
        
        return response
        
    finally:
        # Clean up local PDF if requested
        if cleanup_pdf and pdf_path.exists():
            pdf_path.unlink()

def ocr_by_id_batch(
    arxiv_ids: list[str], 
    pdf_cache_dir: pathlib.Path | None = None,
    cleanup_pdfs: bool = False,
    wait_for_completion: bool = True, 
    poll_interval: int = 10
):
    """
    Batch OCR processing for multiple arXiv papers.
    Downloads PDFs with rate limiting, uploads to Mistral Cloud, then uses batch API.
    
    Args:
        arxiv_ids: List of arXiv IDs to process
        pdf_cache_dir: Directory to cache downloaded PDFs (default: temp dir)
        cleanup_pdfs: Whether to delete PDFs after processing
        wait_for_completion: If True, wait for batch job to complete and return results
        poll_interval: Seconds to wait between status checks (default: 10)
    
    Returns:
        If wait_for_completion=True: dict mapping arxiv_id to OCR response
        If wait_for_completion=False: (batch_job, uploaded_file_ids, pdf_paths) for manual tracking
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    
    # Prepare PDF cache directory
    if pdf_cache_dir is None:
        pdf_cache_dir = pathlib.Path(tempfile.gettempdir()) / "papersys_ocr_cache"
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download PDFs with rate limiting
    logger.info(f"Downloading {len(arxiv_ids)} PDFs from arXiv...")
    pdf_paths = {}
    for i, arxiv_id in enumerate(arxiv_ids):
        pdf_path = pdf_cache_dir / f"{arxiv_id}.pdf"
        
        if not pdf_path.exists():
            # Rate limiting: wait between downloads (except for first)
            if i > 0:
                time.sleep(ARXIV_DOWNLOAD_DELAY)
            
            success = download_arxiv_pdf(arxiv_id, pdf_path)
            if not success:
                logger.warning(f"Skipping {arxiv_id} due to download failure")
                continue
        else:
            logger.debug(f"Using cached PDF for {arxiv_id}")
        
        pdf_paths[arxiv_id] = pdf_path
    
    if not pdf_paths:
        raise RuntimeError("No PDFs downloaded successfully")
    
    logger.info(f"Successfully prepared {len(pdf_paths)} PDFs")
    
    # Upload PDFs to Mistral Cloud
    logger.info(f"Uploading {len(pdf_paths)} PDFs to Mistral Cloud...")
    uploaded_files = {}
    for arxiv_id, pdf_path in pdf_paths.items():
        with open(pdf_path, 'rb') as f:
            uploaded_file = client.files.upload(
                file={
                    "file_name": f"{arxiv_id}.pdf",
                    "content": f,
                },
                purpose="ocr"
            )
        uploaded_files[arxiv_id] = uploaded_file
        logger.debug(f"Uploaded {arxiv_id}: {uploaded_file.id}")
    
    # Get signed URLs and prepare batch requests
    batch_requests = []
    for arxiv_id, uploaded_file in uploaded_files.items():
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        request = {
            "custom_id": arxiv_id,
            "body": {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                "include_image_base64": True
            }
        }
        batch_requests.append(request)
    
    # Create batch JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
        batch_jsonl_path = f.name
    
    try:
        # Upload batch file
        with open(batch_jsonl_path, 'rb') as f:
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
            metadata={"job_type": "arxiv_ocr", "num_papers": len(batch_requests)}
        )
        
        logger.success(f"Batch job created: {created_job.id}")
        
        if not wait_for_completion:
            return created_job, [f.id for f in uploaded_files.values()], pdf_paths
        
        # Wait for completion
        logger.info(f"Processing {len(batch_requests)} papers...")
        
        while True:
            retrieved_job = client.batch.jobs.get(job_id=created_job.id)
            status = retrieved_job.status
            
            logger.debug(f"Batch status: {status}")
            
            if status == "SUCCESS":
                break
            elif status in ["FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"]:
                raise Exception(f"Batch job {status}: {retrieved_job}")
            
            time.sleep(poll_interval)
        
        # Download and parse results
        output_file_stream = client.files.download(file_id=retrieved_job.output_file)
        results_content = output_file_stream.read().decode('utf-8')
        
        results = {}
        for line in results_content.strip().split('\n'):
            if line:
                result = json.loads(line)
                arxiv_id = result['custom_id']
                results[arxiv_id] = result.get('response', {}).get('body', result)
        
        logger.success(f"Batch processing complete! Processed {len(results)} papers.")
        
        # Clean up Mistral Cloud files
        for uploaded_file in uploaded_files.values():
            try:
                client.files.delete(file_id=uploaded_file.id)
            except Exception as e:
                logger.warning(f"Failed to delete file {uploaded_file.id}: {e}")
        
        # Clean up local PDFs if requested
        if cleanup_pdfs:
            for pdf_path in pdf_paths.values():
                try:
                    pdf_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete PDF {pdf_path}: {e}")
        
        return results
        
    finally:
        # Clean up batch JSONL file
        if os.path.exists(batch_jsonl_path):
            os.unlink(batch_jsonl_path)


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
    full_md = "\n\n".join(md_parts)
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

    