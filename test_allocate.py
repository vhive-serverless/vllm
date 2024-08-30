import torch
from torch import Size
import time
import gc

# Allocate a 6GB tensor
# Assuming float16 data type (each element takes 2 bytes)
size_in_gb = 8
element_size = 2  # bytes (for float16)
num_elements = (size_in_gb * 1024**3) // element_size

# Create the tensor
model_weights = torch.empty(num_elements, dtype=torch.float16, device="cuda")

old_block_number = 1889
shape = Size([2,old_block_number,16,128,16])
tensors = []
for i in range(32):
    start = time.time()
    tensor = torch.zeros(shape, pin_memory=False, device="cuda", dtype=torch.float16).cuda(non_blocking=True)
    # torch.cuda.synchronize()
    allocate_latency = time.time() - start
    tensors.append(tensor)

tensors.clear()

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"Available space: {free_mem/(1024**2):.2f}MB")


allocate_latencys = []
block_number = 5198
shape = Size([2,block_number,16,128,16])
torch.cuda.empty_cache()
gc.collect()

for i in range(32):
    start = time.time()
    tensor = torch.zeros(shape, pin_memory=False, device="cuda", dtype=torch.float16)
    allocate_latency = time.time() - start
    allocate_latencys.append(allocate_latency)
    # tensors[i].data = tensor
    tensors.append(tensor)
    torch.cuda.synchronize()
    # tensors.append(tensor)

avg_allocate_latency = sum(allocate_latencys) / len(allocate_latencys)
print(f"average allocate latency: {avg_allocate_latency:.3f}s, sum of allocate latencys: {sum(allocate_latencys):.3f}s")
print(allocate_latencys)
torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"Available space: {free_mem/(1024**2):.2f}MB")