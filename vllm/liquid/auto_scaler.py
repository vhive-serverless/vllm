from vllm.liquid.request import LiquidOutput, LiquidRequest, LiquidType
from vllm.config import LiquidConfig
from typing import List, Tuple, Dict, Optional
import numpy as np
import time

SCALE_OUT_THRESH_MAP = {
    1: 0.9,
    2: 0.98,
    4: 1,
}

SCALE_IN_THRESH = 0.4
SCALE_IN_THRESH_INSTANT = 0.2

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

    def step(self, cache_usage) -> Optional[LiquidRequest]:
        self.cache_usage_records.append(cache_usage) 
        self.tp_level_records.append(self.current_tp_level)
        latest_timestamp = time.time()
        self.timestamp_records.append(latest_timestamp)

        # check if we need to scale out
        liquid_request = None
        # find out all records within scale-out window
        scale_out_records_index_window = []
        for i, ts in enumerate(self.timestamp_records):
            if latest_timestamp - ts < SCALE_OUT_WINDOW:
                scale_out_records_index_window.append(i)
        # find out all records within scale-in window 
        scale_in_records_index_window = []
        for i, ts in enumerate(self.timestamp_records):
            if latest_timestamp - ts < SCALE_IN_WINDOW:
                scale_in_records_index_window.append(i)
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
            if cache_usage < SCALE_IN_THRESH_INSTANT:
                liquid_request = self._scale_in()
                return liquid_request
        return None

    def _scale_in(self) -> Optional[LiquidRequest]:
        if self.current_tp_level == 2:
            self.current_tp_level = 1
            return LiquidRequest(LiquidType.LIQUID_2_1)
        elif self.current_tp_level == 4:
            self.current_tp_level = 2
            return LiquidRequest(LiquidType.LIQUID_4_2)
        else:
            return None
        

    def _scale_out(self) -> Optional[LiquidRequest]:
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

            
                    


