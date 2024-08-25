from vllm.liquid.sharded_parameter import ShardedParameter, QKVShardedParameter
from vllm.liquid.utils import get_tensor_num_bytes
import torch

torch.cuda.empty_cache()
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"Initially, remaining space on GPU 0: {free_memory/(1024**3):.3f} GB")
# Calculate the number of elements needed
num_elements = (2 * 1024 * 1024 * 1024) // 4  # 4 bytes per float32 element

# Allocate a 1GB tensor with float32 data type
tensor = torch.empty(num_elements, dtype=torch.float32, device="cuda")
sharded_param = ShardedParameter(
    data=tensor,
    num_shards=2,
    shard_dim=0,
)
del tensor

# Verify the size of the tensor
print(f"Total bytes: {get_tensor_num_bytes(sharded_param)/(1024**3):.3f} GB")
torch.cuda.empty_cache()
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"After init, remaining space on GPU 0: {free_memory/(1024**3):.3f} GB")
sharded_param.delete_shard(1)
torch.cuda.empty_cache()
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"After delete, remaining space on GPU 0: {free_memory/(1024**3):.3f} GB")

appended_tensor = torch.empty(num_elements // 2, dtype=torch.float32, device="cuda")
torch.cuda.empty_cache()
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"After init appended tensor, remaining space on GPU 0: {free_memory/(1024**3):.3f} GB")

sharded_param.append_shard(1, appended_tensor)
del appended_tensor
torch.cuda.empty_cache()
free_memory, total_memory = torch.cuda.mem_get_info()
print(f"After append the tensor, remaining space on GPU 0: {free_memory/(1024**3):.3f} GB")
