import json
import os
import pathlib
import re
import tempfile
import time
import urllib.parse

import pypandoc
import requests
from datauri import parse as parse_data_uri
from loguru import logger
from mistralai import Mistral
from pypandoc.pandoc_download import download_pandoc

# arXiv rate limiting: 3 seconds between requests to avoid 403/captcha
ARXIV_DOWNLOAD_DELAY = 3.0

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LUA_FILTER_PATH = SCRIPT_ROOT / "scripts" / "strip-wrapper.lua"
AR5IV_USER_AGENT = "PaperSys-ar5iv-prober/0.1 (+paperops@example.com)"
AR5IV_MARKDOWN_FORMAT = "commonmark_x+tex_math_dollars"
AR5IV_MAX_RETRIES = 3
AR5IV_RETRY_DELAY = 2.0
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s+.*?(references?|bibliography)\b.*$"
)


def _fetch_ar5iv_html(
    arxiv_id: str,
    max_retries: int = AR5IV_MAX_RETRIES,
    retry_delay: float = AR5IV_RETRY_DELAY,
) -> tuple[str, str] | tuple[None, None]:
    """Fetch rendered HTML for an arXiv paper via ar5iv."""

    url = f"https://ar5iv.org/html/{arxiv_id}"
    headers = {"User-Agent": AR5IV_USER_AGENT}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=30,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            logger.warning(
                "ar5iv fetch error for {} (attempt {}/{}): {}",
                arxiv_id,
                attempt,
                max_retries,
                exc,
            )
        else:
            final_host = urllib.parse.urlparse(response.url).hostname or ""

            if response.status_code == 200 and "ar5iv" in final_host:
                return response.text, response.url

            if "arxiv" in final_host:
                logger.info(
                    "{} missing on ar5iv (redirected to {})",
                    arxiv_id,
                    response.url,
                )
                return None, None

            logger.debug(
                "Unexpected ar5iv response for {}: status={} host={}",
                arxiv_id,
                response.status_code,
                final_host,
            )

        if attempt < max_retries:
            sleep_for = retry_delay * attempt
            logger.debug(
                "Retrying ar5iv fetch for {} in {:.1f}s",
                arxiv_id,
                sleep_for,
            )
            time.sleep(sleep_for)

    return None, None


def _html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown using the shared Lua filter."""

    extra_args = ["--wrap=none", "--markdown-headings=atx"]
    if LUA_FILTER_PATH.exists():
        extra_args.extend(["--lua-filter", str(LUA_FILTER_PATH)])
    else:
        logger.warning("Lua filter not found at {}, proceeding without it", LUA_FILTER_PATH)
    try:
        return pypandoc.convert_text(
            html,
            to=AR5IV_MARKDOWN_FORMAT,
            format="html",
            extra_args=extra_args,
        )
    except OSError as exc:
        message = str(exc)
        if "No pandoc was found" not in message:
            raise
        logger.info("Pandoc binary missing; downloading via pypandoc...")
        download_pandoc()
        return pypandoc.convert_text(
            html,
            to=AR5IV_MARKDOWN_FORMAT,
            format="html",
            extra_args=extra_args,
        )


def _strip_reference_section(markdown: str) -> str:
    """Remove the References/Bibliography section to avoid redundant OCR."""

    match = REFERENCE_HEADING_RE.search(markdown)
    if not match:
        return markdown
    cutoff = match.start()
    return markdown[:cutoff].rstrip() + "\n"


def _markdown_to_ocr_response(arxiv_id: str, markdown: str, source_url: str | None) -> dict[str, object]:
    """Wrap Markdown content to mimic Mistral OCR response shape."""

    payload = {
        "source": "ar5iv",
        "arxiv_id": arxiv_id,
        "pages": [
            {
                "markdown": markdown,
                "images": [],
            }
        ],
    }
    if source_url:
        payload["source_url"] = source_url
    return payload


def _maybe_fetch_ar5iv_response(arxiv_id: str) -> dict[str, object] | None:
    """Attempt to fetch Markdown from ar5iv; returns OCR-like response or None."""

    html, source_url = _fetch_ar5iv_html(arxiv_id)
    if not html:
        return None

    try:
        markdown = _html_to_markdown(html)
    except (RuntimeError, OSError) as exc:
        logger.warning("Pandoc conversion failed for {}: {}", arxiv_id, exc)
        return None

    markdown = _strip_reference_section(markdown)
    logger.debug("Using ar5iv markdown for {}", arxiv_id)
    return _markdown_to_ocr_response(arxiv_id, markdown, source_url)

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

def ocr_by_id(
    arxiv_id: str,
    pdf_cache_dir: pathlib.Path | None = None,
    cleanup_pdf: bool = False,
    ar5iv: bool = True,
):
    """
    OCR a single arXiv paper by preferring ar5iv HTML when available.
    
    Args:
        arxiv_id: arXiv paper ID
        pdf_cache_dir: Directory to cache downloaded PDFs (default: temp dir)
        cleanup_pdf: Whether to delete the PDF after processing
        ar5iv: If True, attempt ar5iv HTML → Markdown before PDF OCR
    
    Returns:
        OCR response object from Mistral API
    """
    if ar5iv:
        ar5iv_response = _maybe_fetch_ar5iv_response(arxiv_id)
        if ar5iv_response:
            return ar5iv_response

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
    poll_interval: int = 10,
    ar5iv: bool = True,
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
        ar5iv: If True, attempt ar5iv HTML → Markdown shortcut before OCR batch
    
    Returns:
        If wait_for_completion=True: dict mapping arxiv_id to OCR response
        If wait_for_completion=False: (batch_job, uploaded_file_ids, pdf_paths) for manual tracking
    """
    if ar5iv and not wait_for_completion:
        logger.warning("ar5iv shortcut requires wait_for_completion=True; disabling ar5iv path")
        ar5iv = False

    ar5iv_results: dict[str, dict[str, object]] = {}
    pending_ids: list[str] = list(arxiv_ids)

    if ar5iv:
        pending_ids = []
        for arxiv_id in arxiv_ids:
            ar5iv_response = _maybe_fetch_ar5iv_response(arxiv_id)
            if ar5iv_response:
                ar5iv_results[arxiv_id] = ar5iv_response
            else:
                pending_ids.append(arxiv_id)

    if not pending_ids:
        logger.info(
            "ar5iv hits: %d/%d (skipped OCR batch)",
            len(ar5iv_results),
            len(arxiv_ids),
        )
        return ar5iv_results

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    
    # Prepare PDF cache directory
    if pdf_cache_dir is None:
        pdf_cache_dir = pathlib.Path(tempfile.gettempdir()) / "papersys_ocr_cache"
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download PDFs with rate limiting
    logger.info(f"Downloading {len(pending_ids)} PDFs from arXiv...")
    pdf_paths = {}
    for i, arxiv_id in enumerate(pending_ids):
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
        
        logger.info(
            "ar5iv hits: %d/%d (remaining %d processed via OCR)",
            len(ar5iv_results),
            len(arxiv_ids),
            len(results),
        )
        combined_results = {**ar5iv_results, **results}
        return combined_results
        
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
    
    logger.info(
        "Markdown saved to %s (pages=%d, images=%d)",
        md_path,
        num_pages,
        num_images,
    )

    
