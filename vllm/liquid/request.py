from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class LiquidRequest:
    '''Request to perform liquid operation, will be sent to scheduler for scheduling
    '''
    shard_ids: List[int]
    src: int
    dst: int

@dataclass
class LiquidOutput:
    shard_ids: List[int]
    src: int
    dst: int
    
    freed_memory_GB: Optional[float] = None
    liquid_e2e_latency: Optional[float] = None

    def __repr__(self) -> str:
        repr = f"Move shard: {self.shard_ids} from {self.src} to {self.dst}"
        if self.freed_memory_GB is not None and self.liquid_e2e_latency is not None:
            repr += f"e2e_latency: {self.liquid_e2e_latency:.2f} s; freed memory: {self.freed_memory_GB:.2f} GB"

        return repr
