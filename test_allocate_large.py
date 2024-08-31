import torch
import time

size_in_gb = 13
element_size = 2  # bytes (for float16)
num_elements = (size_in_gb * 1024**3) // element_size

#pre-warm
# torch.empty(num_elements, dtype=torch.float16, device="cuda")

# Create the tensor
start = time.time()
large_tensor = torch.empty(num_elements, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
allocate_latency = time.time() - start
print(f"allocate a {size_in_gb:.1f}GB tensor takes: {allocate_latency:.3f}s")

free_mem, _ = torch.cuda.mem_get_info()
print(f"After allocating tensor, free space: {free_mem/(1024**3):.1f}GB")

# large_tensor.data = torch.empty(0)
del large_tensor
# torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"After free its data, free space: {free_mem/(1024**3):.1f}GB")

# large_tensor.data = torch.empty(num_elements, dtype=torch.float16, device="cuda")
# torch.cuda.empty_cache()
# free_mem, _ = torch.cuda.mem_get_info()
# print(f"After allocate again, free space: {free_mem/(1024**3):.1f}GB")
