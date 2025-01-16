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

    def __repr__(self) -> str:
        liquid_type_str = ""
        if self.liquid_type == LiquidType.LIQUID_1_2:
            liquid_type_str = "Scale out from GPU0 to GPU1"
        elif self.liquid_type == LiquidType.LIQUID_2_4:
            liquid_type_str = "Scale out from GPU[0,1] to GPU[2,3]"
        elif self.liquid_type == LiquidType.LIQUID_4_2:
            liquid_type_str = "Scale in from GPU[2,3] to GPU[0,1]"
        elif self.liquid_type == LiquidType.LIQUID_2_1:
            liquid_type_str = "Scale in from GPU1 to GPU 0"
        return liquid_type_str

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
    srcs: List[int]
    dsts: List[int]
    is_scale_out: bool

    # when scale in, need to move blocks, below is the mapping of src block number to dst block number
    src_to_dsts: List[Tuple[int,int]] = field(default_factory=list)
    
    freed_memory_GB: Optional[float] = None
    liquid_e2e_latency: Optional[float] = None

    #timestamps:
    liquid_start: float = 0
    finished_move_and_shrink = 0
    finished_update_workers: float = 0
    finished_liquid_model_weights: float= 0
    finished_init_mem: float = 0
    finished_liquid_kvc: float = 0
    finished_extending_gpu_blocks: float = 0
    finished_update_blocks: float = 0


    def __repr__(self) -> str:
        repr = f"Completed! Move shard: {self.shard_ids} from {self.srcs} to {self.dsts};"
        if self.freed_memory_GB is not None and self.liquid_e2e_latency is not None:
            repr += f"e2e_latency: {self.liquid_e2e_latency:.2f} s; freed memory: {self.freed_memory_GB:.2f} GB"

        
        e2e_latency = self.finished_update_blocks - self.liquid_start
        repr += f"liquid e2e latency: {e2e_latency:.2f}s, "
        if not self.is_scale_out:
            move_and_shrink_latency = self.finished_move_and_shrink - self.liquid_start     
            repr += f"move and shrink latency: {move_and_shrink_latency:.2f}s, "
            update_worker_latency = self.finished_update_workers - self.finished_move_and_shrink
        else:
            update_worker_latency = self.finished_update_workers - self.liquid_start
        repr += f"update worker latency: {update_worker_latency:.2f}s, "

        liquid_model_weights_latency = self.finished_liquid_model_weights - self.finished_update_workers
        repr += f"liquid model weights latency: {liquid_model_weights_latency:.2f}s, "
        init_mem_latency = self.finished_init_mem - self.finished_liquid_model_weights
        repr += f"init mem latency: {init_mem_latency:.2f}s, "
        liquid_kvc_latency = self.finished_liquid_kvc - self.finished_init_mem
        repr += f"liquid kvc latency: {liquid_kvc_latency:.2f}s, "

        if self.is_scale_out:
            extending_gpu_blocks_latency = self.finished_extending_gpu_blocks - self.finished_liquid_kvc
            repr += f"extending gpu blocks latency: {extending_gpu_blocks_latency:.2f}s, "
            update_blocks_latency = self.finished_update_blocks - self.finished_extending_gpu_blocks
        else:
            update_blocks_latency = self.finished_update_blocks - self.finished_liquid_kvc
        repr += f"update blocks latency: {update_blocks_latency:.2f}s;"


        return repr
