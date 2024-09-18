import os
import torch
import torch.distributed as dist
import time
import re
from typing import Dict, List, Any
import subprocess
# Sender side functions
def _encode_dict_to_tensors(tensor_dict, dtype=torch.float16):
    meta_data = {}
    data_list = []
    start_idx = 0

    for key, tensor in tensor_dict.items():
        length = tensor.numel()
        meta_data[key] = (start_idx, length, tensor.shape, tensor.dtype)
        data_list.append(tensor.flatten())
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
    print(f"encode latency: {latency:.2f} s")
    
    # Store meta data in TCPStore
    start = time.time()
    _store_meta_data(meta_data, store)
    latency = time.time() - start
    print(f"store latency: {latency:.2f} s")

    dist.barrier() 
    start = time.time()
    dist.send(data_tensor, dst=dst_rank, group=group)
    dist.barrier()
    print(f"sending latency: {time.time() - start:.2f}s")

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
    dist.barrier()
    dist.recv(data_tensor, src=src_rank, group=group)
    dist.barrier()

    start = time.time() 
    tensor_dict = _decode_tensors_to_dict(meta_data, data_tensor)
    latency = time.time() - start
    return tensor_dict

def get_tensor_num_bytes(tensor: torch.Tensor) -> int:

    num_elements = tensor.numel()

    # Get the number of bits per element
    bits = torch.finfo(tensor.dtype).bits  # or torch.iinfo(tensor.dtype).bits for integers

    # Calculate the total memory occupied (in bytes)
    total_memory_bytes = (num_elements * bits) // 8
    return total_memory_bytes

DEBUG_MODE = True

def get_cuda_mem_info(device: int=0) -> str:
        torch.cuda.set_deivce(f"cuda:{device}")
        if DEBUG_MODE:
            torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info()
        allocated_space = torch.cuda.memory_allocated()
        researved_space = torch.cuda.memory_reserved()
     
        return f"allocated space on GPU: {allocated_space/(1024**3):.3f} GB, reserved space on GPU: {researved_space/(1024**3):.3f} GB, free space: {free_mem/(1024**3):.3f}GB, frag space: {(researved_space - allocated_space)/(1024**3):.3f}GB"

def get_gpu_processes_and_memory(gpu_id=0):
    # Run the nvidia-smi command to get the details of the processes on a specific GPU
    command = f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i {gpu_id}"
    
    # Execute the command and capture the output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Error running nvidia-smi: {result.stderr.decode('utf-8')}")
        return None
    
    # Parse the result
    output = result.stdout.decode('utf-8').strip()
    
    # Check if there are no processes using the GPU
    if not output:
        print(f"No processes are using GPU {gpu_id}")
        return None
    
    # Split output into lines and extract PID and memory usage
    process_info = []
    for line in output.split('\n'):
        pid, memory_used = line.split(',')
        memory_used = float(memory_used.strip()) / (1024)
        process_info.append((int(pid.strip()), memory_used))  # Convert to integers

    return process_info
