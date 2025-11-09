#!/usr/bin/env python3
"""
Test script for batch OCR processing with PDF download.
Usage: uv run python scripts/test_ocr_batch.py
"""

import os
import pathlib
from loguru import logger
from papersys.ocr import ocr_by_id_batch, response2md

def main():
    # Test with a few papers
    test_ids = [
        "2201.04234",  # Example paper 1
        "2410.12613",  # Example paper 2 (already in your example/)
        "2505.11739",  # Example paper 3 (already in your example/)
    ]
    
    logger.info(f"Testing batch OCR with {len(test_ids)} papers")
    
    # Set up cache directory
    pdf_cache_dir = pathlib.Path("data/pdf_cache")
    output_dir = pathlib.Path("data/ocr_responses")
    
    try:
        # Run batch OCR
        results = ocr_by_id_batch(
            arxiv_ids=test_ids,
            pdf_cache_dir=pdf_cache_dir,
            cleanup_pdfs=False,  # Keep PDFs for inspection
            wait_for_completion=True,
            poll_interval=10
        )
        
        logger.info(f"Got results for {len(results)} papers")
        
        # Convert to markdown
        for arxiv_id, ocr_response in results.items():
            try:
                md_path, img_dir, num_pages, num_images = response2md(
                    ocr_response=ocr_response,
                    output_dir=output_dir / arxiv_id,
                    filename=arxiv_id
                )
                logger.success(f"Saved {arxiv_id}: {num_pages} pages, {num_images} images")
            except Exception as e:
                logger.error(f"Failed to save {arxiv_id}: {e}")
        
        logger.success("Test completed successfully!")
        logger.info(f"PDFs cached in: {pdf_cache_dir}")
        logger.info(f"Markdown output in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
