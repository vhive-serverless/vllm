from typing import List, Optional, Tuple, Dict
import torch
Slice = Tuple[int, int]
LiquidConfigType = Dict[Slice, torch.device]

LIQUIDCONFIG : LiquidConfigType = {
    (0,1):torch.device(0),
    (1,2):torch.device(1)
}

LIQUID_DEVICE_SET = [torch.cuda.device(0), torch.cuda.device(1)]
# LIQUID_DEVICE_SET = [torch.cuda.device(1)]