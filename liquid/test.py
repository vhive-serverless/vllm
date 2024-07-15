from sharded_parameter import ShardedParameter
from sharded_tensor import ShardedTensor
import torch

def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3

# Main script for testing
if __name__ == "__main__":
    # Create a tensor of size 4000 * 6000
    tensor_size = (4000, 6000)
    num_shards = 4
    shard_size = tensor_size[1] // num_shards

    # Initialize the sharded tensor with the specified values
    data = torch.cat([
        torch.full((tensor_size[0], shard_size), 0, dtype=torch.float32, device='cuda'),
        torch.full((tensor_size[0], shard_size), 1, dtype=torch.float32, device='cuda'),
        torch.full((tensor_size[0], shard_size), 2, dtype=torch.float32, device='cuda'),
        torch.full((tensor_size[0], shard_size), 3, dtype=torch.float32, device='cuda')
    ], dim=1)

    memory_usage = get_gpu_memory_usage()
    print(f"initial_memory_usage: {memory_usage:.3f} GB")
    # # Create a ShardedParameter instance
    sharded_param = ShardedTensor(data=data, num_shards=num_shards, shard_dim=1)
    # clone_data = data.clone()


    del data

    # Delete shard 1 (second shard, full of 1) and shard 3 (fourth shard, full of 3)
    sharded_param.delete_shard(1)
    sharded_param.delete_shard(3)  

    # Get the second shard 
    remaining_shard = sharded_param.get_shard(2)

    memory_usage = get_gpu_memory_usage()
    print(f"current_memory_usage: {memory_usage:.3f} GB")

    # Check if the remaining shard is still full of 2
    assert sharded_param.shape == (4000, 3000)
    assert torch.all(remaining_shard == 2), "The remaining shard is not full of 2"

    print("Test passed: The remaining shard is still full of 2")