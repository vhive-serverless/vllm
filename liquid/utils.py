
import torch
def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3