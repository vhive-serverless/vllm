from typing import Dict, List, Any, Tuple
from attr import dataclass

@dataclass
class PlaceRequest:
    layer_range: Tuple[int, int]
    dest_gpu_id: int