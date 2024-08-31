import torch

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"initially, free mem: {free_mem/(1024**3):.3f}GB")

q = torch.empty(768, 4096, device="cuda")
k = torch.empty(768, 4096, device="cuda")
v = torch.empty(768, 4096, device="cuda")

torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
print(f"after qkv, free mem: {free_mem/(1024**3):.3f}GB, each of the tensor has: {q.numel()} elemented")

nested_tensor = torch.nested.nested_tensor([q,k,v])
torch.cuda.empty_cache()
free_mem, _ = torch.cuda.mem_get_info()
numel = nested_tensor.numel()
print(f"after creating nested tensor, free mem: {free_mem/(1024**3):.3f}GB, nested tensor has {nested_tensor.numel()} elements")

flatten_tensor = nested_tensor.view(3, numel//3)
print(flatten_tensor[0].shape)