import os
import torch
import torch.distributed as dist
import time
import re
from typing import Dict, List, Any
# Sender side functions
def _encode_dict_to_tensors(tensor_dict, dtype=torch.float16):
    meta_data = {}
    data_list = []
    start_idx = 0

    for key, tensor in tensor_dict.items():
        length = tensor.numel()
        meta_data[key] = (start_idx, length, tensor.shape, tensor.dtype)
        data_list.append(tensor.flatten().to(device="cuda", dtype=dtype))
        start_idx += length
    
    data_tensor = torch.cat(data_list).cuda()
    return meta_data, data_tensor

def _store_meta_data(meta_data, store):
    for key, (start_idx, length, shape, dtype) in meta_data.items():
        store.set(f"{key}_start_idx", str(start_idx))
        store.set(f"{key}_length", str(length))
        store.set(f"{key}_shape", str(shape))
        store.set(f"{key}_dtype", str(dtype))

def send_dict(tensor_dict: Dict[str, torch.Tensor], dst_rank: int, store: Any, group: Any = None, dtype=torch.float16):
    start = time.time()
    meta_data, data_tensor = _encode_dict_to_tensors(tensor_dict, dtype)
    latency = time.time() - start
    # print(f"encode latency: {latency:.2f} s, data_tensor: {data_tensor}")
    
    # Store meta data in TCPStore
    start = time.time()
    _store_meta_data(meta_data, store)
    latency = time.time() - start
    # print(f"store latency: {latency:.2f} s")
    
    dist.send(data_tensor, dst=dst_rank, group=group)

# Receiver side functions
def _str_to_dtype(dtype_str):
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float': torch.float,
        'torch.float64': torch.float64,
        'torch.double': torch.double,
        'torch.float16': torch.float16,
        'torch.half': torch.half,
        'torch.int64': torch.int64,
        'torch.long': torch.long,
        'torch.int32': torch.int32,
        'torch.int': torch.int,
        'torch.int16': torch.int16,
        'torch.short': torch.short,
        'torch.int8': torch.int8,
        'torch.uint8': torch.uint8,
        'torch.bool': torch.bool,
    }
    return dtype_map[dtype_str]

def _extract_size_from_string(size_str):
    # Use regular expression to find the content inside parentheses
    match = re.search(r'\[([0-9, ]+)\]', size_str)
    if match:
        # Extract the matched content and split by comma to get the dimensions
        size_str = match.group(1)
        size_list = [int(dim) for dim in size_str.split(',')]
        return size_list
    else:
        raise ValueError(f"No size found in the string: {size_str}")

def _retrieve_meta_data(store, keys):
    meta_data = {}
    for key in keys:
        start_idx = int(store.get(f"{key}_start_idx"))
        length = int(store.get(f"{key}_length"))
        shape_str = store.get(f"{key}_shape").decode()
        shape = tuple(_extract_size_from_string(shape_str))
        dtype_str = store.get(f"{key}_dtype").decode()
        dtype = _str_to_dtype(dtype_str)
        meta_data[key] = (start_idx, length, shape, dtype)
    return meta_data

def _decode_tensors_to_dict(meta_data, data_tensor):
    tensor_dict = {}
    
    for key, (start_idx, length, shape, dtype) in meta_data.items():
        tensor = data_tensor[start_idx:start_idx + length].view(shape).to(dtype)
        tensor_dict[key] = tensor
    
    return tensor_dict

def receive_dict(src_rank: int, store: Any, keys: List[str], group = None, dtype=torch.float16):
    
    # Retrieve the list of keys from the sender (you can store these in TCPStore too if needed)# Example, adapt as needed
    start = time.time()
    meta_data = _retrieve_meta_data(store, keys)
    latency = time.time() - start
    print(f"retireve latency: {latency:.2f} s")
    
    total_length = sum(length for _, length, _, _ in meta_data.values())
    # total_length = 77425920
    data_tensor = torch.empty(total_length, device='cuda', dtype=dtype)


    
    # Receive the data tensor
    # dist.barrier()
    dist.recv(data_tensor, src=src_rank, group=group)
    # dist.barrier(group=group)

    start = time.time() 
    tensor_dict = _decode_tensors_to_dict(meta_data, data_tensor)
    latency = time.time() - start
    return tensor_dict