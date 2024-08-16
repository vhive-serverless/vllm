from typing import Dict, List, Any, Set
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper)
class LiquidWorkerInfo:
    def __init__(self, worker: ProcessWorkerWrapper, rank: int, shard_ids: List[int], is_active: bool, initialized: bool) -> None:
        self.worker = worker
        self.rank = rank
        self.shard_ids = set(shard_ids)
        self.is_active = is_active
        self.initialized = initialized

    def set_active(self, is_active: bool):
        self.is_active = is_active
        if self.rank != 0:
            self.worker.is_active = is_active