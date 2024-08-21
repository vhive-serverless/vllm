from vllm.liquid.sharded_parameter import ShardedParameter
import torch
from torch import Tensor
import gc
def get_gpu_memory_usage_in_gb(device=0):
    memory_allocated = torch.cuda.memory_allocated(device)
    
    memory_in_gb = memory_allocated / (1024 ** 3)
    
    return memory_in_gb

dim1 = 65536  # For example, 65536 rows
dim2 = 4096   # For example, 4096 columns

tensor = ShardedParameter(
    data=torch.empty(dim1, dim2, dtype=torch.float32, device="cuda:0"),
    num_shards=2,
    shard_dim=0,
)
memory_usage = get_gpu_memory_usage_in_gb()
print(f"Current memory usage: {memory_usage:.1f} GB")
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"There is still {free_memory/(1024**3):.2f} GB")

tensor.delete_shard(1)
# torch.cuda.empty_cache()
# gc.collect()

memory_usage = get_gpu_memory_usage_in_gb()
print(f"Current memory usage: {memory_usage:.1f} GB")
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"There is still {free_memory/(1024**3):.2f} GB")


tensor.delete_shard(0)
memory_usage = get_gpu_memory_usage_in_gb()
print(f"Current memory usage: {memory_usage:.1f} GB")

data=torch.empty(dim1 // 2, dim2, dtype=torch.float32, device="cuda:0")
tensor.append_shard(3, data)
del data
memory_usage = get_gpu_memory_usage_in_gb()
print(f"Current memory usage: {memory_usage:.1f} GB")

