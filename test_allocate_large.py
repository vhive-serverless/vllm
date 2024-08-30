import torch
import time

size_in_gb = 20
element_size = 2  # bytes (for float16)
num_elements = (size_in_gb * 1024**3) // element_size

#pre-warm
torch.empty(num_elements, dtype=torch.float16, device="cuda")

# Create the tensor
start = time.time()
large_tensor = torch.empty(num_elements, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
allocate_latency = time.time() - start
print(f"allocate a {size_in_gb:.1f}GB tensor takes: {allocate_latency:.3f}s")

# start = time.time()
# del large_tensor
# torch.cuda.synchronize()
# delete_latency = time.time() - start
# print(f"delete the tensor takes: {delete_latency:.2f}s")

start = time.time()
large_tensor = torch.empty(num_elements // 2, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
allocate_latency = time.time() - start
print(f"allocate the large tensor again takes: {allocate_latency:.3f}s")