import torch
element_size = torch.tensor([], dtype=self.dtype, device="cuda:0").element_size()
print(f"on cuda:0, element_size: {element_size}")