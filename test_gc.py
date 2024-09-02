import torch
# Allocate a 6GB tensor
# Assuming float16 data type (each element takes 2 bytes)
size_in_gb = 1
element_size = 2  # bytes (for float16)
num_elements = (size_in_gb * 1024**3) // element_size

# Create the tensor
kv_caches = []

for i in range(12):
    kv_caches.append(torch.empty(num_elements, dtype=torch.float16, device="cuda"))

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"free mem: {free_mem/(1024**3):.2f}GB")

for i in range(12):
    kv_caches[i].data = torch.empty(num_elements//2, dtype=torch.float16, device="cuda")

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"free mem: {free_mem/(1024**3):.2f}GB")

for i in range(12):
    kv_caches[i].data = torch.empty(0, dtype=torch.float16, device="cuda")

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"free mem: {free_mem/(1024**3):.2f}GB")
