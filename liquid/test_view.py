from sharded_parameter import ShardedParameter
from sharded_tensor import ShardedTensor
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from liquid.utils import get_gpu_memory_usage

# Main script for testing
if __name__ == "__main__":
    # Create a tensor of size 4000 * 6000
    tensor_size = 24000000
    num_shards = 4
    shard_size = tensor_size // num_shards

    # tensor = torch.zeros(tensor_size, device='cuda')
    # memory_usage = get_gpu_memory_usage()
    # print(f"initial_memory_usage: {memory_usage:.3f} GB")

    # view = tensor.view(4000,-1)
    # print(f"view's shape: {view.shape}")
    # memory_usage = get_gpu_memory_usage()
    # print(f"after creating view: {memory_usage:.3f} GB")

    sharded_tensor = ShardedTensor(torch.zeros(tensor_size, device='cuda').view(4000,-1), num_shards=num_shards, shard_dim=-1)
    memory_usage = get_gpu_memory_usage()
    print(f"after creating sharded_tensor: {memory_usage:.3f} GB")

    sharded_tensor.delete_shard(0)
    memory_usage = get_gpu_memory_usage()
    print(f"after delete one sharded_tensor: {memory_usage:.3f} GB")
    print(f"sharded_tensor's shape: {sharded_tensor.shape}")

    