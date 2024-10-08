from vllm.liquid.request import LiquidOutput, LiquidRequest, LiquidType
from vllm.config import LiquidConfig, CacheConfig
from typing import List, Tuple, Dict, Optional
import numpy as np
import time
import os

SCALE_OUT_THRESH = float(os.getenv("SCALE_OUT_THRESH", 3))
SCALE_IN_THRESH = float(os.getenv("SCALE_IN_THRESH", 0.7))
# SCALE_OUT_THRESH = 3
SCALE_OUT_THRESH_MAP = {
    1: SCALE_OUT_THRESH,
    2: SCALE_OUT_THRESH,
    4: SCALE_OUT_THRESH,
}

# SCALE_IN_THRESH = 0.7

SCALE_OUT_WINDOW = 15
SCALE_IN_WINDOW = 5

class AutoScaler:
    def __init__(self, liquid_config: LiquidConfig) -> None:
        self.current_tp_level = 1
        self.liquid_config = liquid_config
        self.liquid_gpu_num = len(liquid_config.liquid_gpu_range)
        # cache util records will keep util records for 30 seconds
        self.cache_usage_records : List[float] = []
        self.tp_level_records: List[int] = []
        # These two metrics will be updated every time window
        self.timestamp_records: List[float] = []

        self.bsz_records: List[int] = []

        self.num_gpu_blocks_stack: List[int] = []

        self.is_full_flag: bool = False # The flag is set to True when this instance has reached its liquid gpu range but still wants to scale out, will be set to False again when it is reported to global scaler
        self.latest_num_concurrent_blocks = 0

    def step(self, concurrent_cache_usage,num_concurrent_blocks, num_using_gpu_blocks, metrics) -> Optional[LiquidRequest]:
        self.latest_num_concurrent_blocks = num_concurrent_blocks
        self.cache_usage_records.append(concurrent_cache_usage) 
        self.tp_level_records.append(self.current_tp_level)
        self.bsz_records.append(metrics.num_running)
        latest_timestamp = time.time()
        self.timestamp_records.append(latest_timestamp)

        # check if we need to scale out
        liquid_request = None
        # find out all records within scale-out window
        scale_out_records_index_window = []
        scale_in_records_index_window = []
        # iterate the list in reverse order to avoid iterating too many elements
        for i in range(len(self.timestamp_records)):
            index = (len(self.timestamp_records) - 1) - i
            ts = self.timestamp_records[index]
            if latest_timestamp - ts < SCALE_OUT_WINDOW:
                scale_out_records_index_window.append(index)
            else:
                break
        # find out all records within scale-in window 
        for i in range(len(self.timestamp_records)):
            index = (len(self.timestamp_records) - 1) - i
            ts = self.timestamp_records[index]
            if latest_timestamp - ts < SCALE_IN_WINDOW:
                scale_in_records_index_window.append(index)
            else:
                break
        # If the time window only contains one element, do not scale
        cache_usages = [self.cache_usage_records[i] for i in scale_out_records_index_window]
        cache_usages = np.array(cache_usages)
        mean_value = np.mean(cache_usages)
        if mean_value > SCALE_OUT_THRESH_MAP[self.current_tp_level]:
            liquid_request = self._scale_out()
            return liquid_request

        cache_usages = [self.cache_usage_records[i] for i in scale_in_records_index_window]
        cache_usages = np.array(cache_usages)

        mean_value = np.mean(cache_usages)

        if mean_value < SCALE_IN_THRESH:
            if len(self.num_gpu_blocks_stack) < 2:
                return None

            if num_using_gpu_blocks < self.num_gpu_blocks_stack[-2]:
                print(f"Currently using blocks: #{num_using_gpu_blocks} < previous gpu block: #{self.num_gpu_blocks_stack[-2]} ,decide to scale in!")
                liquid_request = self._scale_in()
                return liquid_request
            else:
                print(f"Try to scale in but currently using blocks #{num_using_gpu_blocks} > previous gpu blocks: {self.num_gpu_blocks_stack[-2]}")
        return None

    def _scale_in(self) -> Optional[LiquidRequest]:
        # TODO: when current tp level is 1 and still try to scale in, this should also be reported to global scaler
        if self.current_tp_level == 2:
            self.current_tp_level = 1
            return LiquidRequest(LiquidType.LIQUID_2_1)
        elif self.current_tp_level == 4:
            self.current_tp_level = 2
            return LiquidRequest(LiquidType.LIQUID_4_2)
        else:
            return None
        

    def _scale_out(self) -> Optional[LiquidRequest]:
        if self.current_tp_level >= self.liquid_gpu_num:
            self.is_full_flag = True

        if self.current_tp_level == 1:
            self.current_tp_level = 2
            return LiquidRequest(LiquidType.LIQUID_1_2)
        elif self.current_tp_level == 2:
            if self.liquid_gpu_num < 4:
                return None
            self.current_tp_level = 4
            return LiquidRequest(LiquidType.LIQUID_2_4)
        else:
            return None

            
                    


