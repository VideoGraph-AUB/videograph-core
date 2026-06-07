"""
Parallel processing module for video ingestion.

Enables concurrent LLM calls for:
- Visual captioning (multiple clips in parallel)
- OCR extraction (multiple frames in parallel)
- Embedding computation (batch processing)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingProgress:
    """Progress information for a processing task."""
    stage: str
    current: int
    total: int
    message: str
    elapsed_seconds: float = 0.0
    
    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


class ParallelProcessor:
    """
    Manages parallel execution of LLM tasks with rate limiting.
    
    Uses ThreadPoolExecutor for I/O-bound OpenAI API calls.
    Includes built-in rate limiting to avoid API throttling.
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        rate_limit_rpm: int = 60,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum concurrent workers (OpenAI recommends <= 5 for Vision)
            rate_limit_rpm: Requests per minute limit
            progress_callback: Optional callback for progress updates
        """
        self.max_workers = max_workers
        self.rate_limit_rpm = rate_limit_rpm
        self.progress_callback = progress_callback
        self._request_times: List[float] = []
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        # Remove requests older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rate_limit_rpm:
            # Wait until the oldest request is 60+ seconds old
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self._request_times.append(time.time())
    
    def _report_progress(self, stage: str, current: int, total: int, message: str, start_time: float):
        """Report progress if callback is set."""
        if self.progress_callback:
            progress = ProcessingProgress(
                stage=stage,
                current=current,
                total=total,
                message=message,
                elapsed_seconds=time.time() - start_time
            )
            self.progress_callback(progress)
    
    def process_parallel(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any],
        stage_name: str = "processing",
        item_name: str = "item"
    ) -> List[Any]:
        """
        Process items in parallel with progress tracking.
        
        Args:
            items: List of items to process
            process_fn: Function to apply to each item
            stage_name: Name of the processing stage (for logging)
            item_name: Name of item type (for logging)
            
        Returns:
            List of results in same order as input items
        """
        if not items:
            return []
        
        total = len(items)
        results = [None] * total
        completed = 0
        start_time = time.time()
        
        logger.info(f"[{stage_name}] Starting parallel processing of {total} {item_name}s with {self.max_workers} workers")
        self._report_progress(stage_name, 0, total, f"Starting {stage_name}...", start_time)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, item in enumerate(items):
                self._wait_for_rate_limit()
                future = executor.submit(process_fn, item)
                future_to_idx[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    
                    msg = f"Processed {completed}/{total} {item_name}s ({rate:.1f}/s, ETA: {eta:.0f}s)"
                    logger.info(f"[{stage_name}] {msg}")
                    self._report_progress(stage_name, completed, total, msg, start_time)
                    
                except Exception as e:
                    logger.error(f"[{stage_name}] Failed to process {item_name} {idx}: {e}")
                    results[idx] = None
        
        elapsed = time.time() - start_time
        logger.info(f"[{stage_name}] Completed {total} {item_name}s in {elapsed:.1f}s ({total/elapsed:.1f}/s)")
        self._report_progress(stage_name, total, total, f"Completed {stage_name}", start_time)
        
        return results


async def process_parallel_async(
    items: List[Any],
    process_fn: Callable[[Any], Any],
    max_concurrent: int = 5,
    stage_name: str = "processing"
) -> List[Any]:
    """
    Async version of parallel processing using asyncio.
    
    Useful when called from async context.
    
    Args:
        items: List of items to process
        process_fn: Sync function to apply (will be run in thread pool)
        max_concurrent: Maximum concurrent tasks
        stage_name: Name for logging
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    semaphore = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_event_loop()
    
    async def process_with_semaphore(item, idx):
        async with semaphore:
            return await loop.run_in_executor(None, process_fn, item)
    
    tasks = [process_with_semaphore(item, idx) for idx, item in enumerate(items)]
    
    logger.info(f"[{stage_name}] Starting async processing of {len(items)} items")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"[{stage_name}] Item {idx} failed: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)
    
    logger.info(f"[{stage_name}] Completed processing {len(items)} items")
    return processed_results


