import torch

# Function to get the current GPU memory usage
def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB

# 1. Allocate a 1-dimensional tensor that occupies ~1 GB of GPU memory
# Assuming we use float32 (4 bytes per element), calculate the size
size_in_gb = 20
element_size = 4  # float32 is 4 bytes
num_elements = (size_in_gb * 1024 ** 3) // element_size

# Allocate the tensor
large_tensor = torch.empty(num_elements, dtype=torch.float32, device='cuda')
size_in_gb = 10
element_size = 4  # float32 is 4 bytes
num_elements = (size_in_gb * 1024 ** 3) // element_size
small_tensor = torch.empty(num_elements, dtype=torch.float32, device='cuda')

large_tensor = torch.cat([large_tensor, small_tensor])

# Get the GPU memory after allocation
initial_memory = get_gpu_memory_usage()
print(f"GPU memory after tensor allocation: {initial_memory:.2f} GB")


# Get the GPU memory after narrowing
final_memory = get_gpu_memory_usage()
print(f"GPU memory after narrowing tensor: {final_memory:.2f} GB")