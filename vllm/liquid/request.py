from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

class LiquidType(Enum):
    LIQUID_1_2 = 1
    LIQUID_2_4 = 2
    LIQUID_2_1 = 3
    LIQUID_4_2 = 4


@dataclass
class LiquidRequest:
    liquid_type: LiquidType

# @dataclass
# class LiquidRequest:
#     '''Request to perform liquid operation, will be sent to scheduler for scheduling
#     '''
#     shard_ids: List[int]
#     src: int
#     dst: int
#     is_scale_out: bool

@dataclass
class LiquidOutput:
    shard_ids: List[int]
    src: int
    dst: int

    # when scale in, need to move blocks, below is the mapping of src block number to dst block number
    src_to_dsts: List[Tuple[int,int]] = field(default_factory=list)
    
    freed_memory_GB: Optional[float] = None
    liquid_e2e_latency: Optional[float] = None

    #timestamps:
    liquid_start: float = 0
    finished_update_workers: float = 0
    finished_liquid_model_weights: float= 0
    finished_init_mem: float = 0
    finished_liquid_kvc: float = 0

    def __repr__(self) -> str:
        repr = f"Completed! Move shard: {self.shard_ids} from {self.src} to {self.dst};"
        if self.freed_memory_GB is not None and self.liquid_e2e_latency is not None:
            repr += f"e2e_latency: {self.liquid_e2e_latency:.2f} s; freed memory: {self.freed_memory_GB:.2f} GB"

        e2e_latency = self.finished_liquid_kvc - self.liquid_start
        update_worker_latency = self.finished_update_workers - self.liquid_start
        liquid_model_weights_latency = self.finished_liquid_model_weights - self.finished_update_workers
        init_mem_latency = self.finished_init_mem - self.finished_liquid_model_weights
        liquid_kvc_latency = self.finished_liquid_kvc - self.finished_init_mem
        repr += f"liquid e2e latency: {e2e_latency:.2f}s, update worker latency: {update_worker_latency:.2f}s, liquid model weights latency: {liquid_model_weights_latency:.2f}s, init mem latency: {init_mem_latency:.2f}s, liquid kvc latency: {liquid_kvc_latency:.2f}s;"

        return repr
