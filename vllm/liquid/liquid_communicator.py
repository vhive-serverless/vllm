import torch
import torch.distributed as dist
from typing import Dict, Tuple, List
from vllm.liquid.metadata import MetaData
import math
import time

class LiquidCommunicator:
    def __init__(self, buffer_size_gb: float, group:dist.group, tcp_store_port: int = 12345, dtype=torch.float16):
        self.buffer_size_gb = buffer_size_gb
        self.buffer_length = int(buffer_size_gb * (1024**3) / torch.finfo(dtype).bits * 8)  # Calculate number of elements
        self.dtype = dtype

        # Initialize buffer
        self.buffer = torch.empty(self.buffer_length, dtype=dtype, device='cuda')

        # Initialize process group
        self.group = group

        self.rank = dist.get_rank(self.group) 
        world_size = dist.get_world_size(self.group)
        # Initialize TCP Store
        self.store = dist.TCPStore("localhost", tcp_store_port, world_size=world_size, is_master=(self.rank == 0))
        self.meta_keys: List[str] = []

    def _send_meta_data(self, tensor_dict: Dict[str, torch.Tensor]):
        offset = 0
        for key, tensor in tensor_dict.items():
            shape = tensor.shape
            length = tensor.flatten().numel()
            meta_data = MetaData(offset, length, shape)
            self.store.set(key, str(meta_data))
            self.meta_keys.append(key)
            offset += length
            
        # Send the metadata using TCPStore
        self.store.set(f"total_length", str(offset))
        self.meta_keys.append("total_length")

    def _clear_store(self):
        for meta_key in self.meta_keys:
            success = self.store.delete_key(meta_key)
            assert success
        self.meta_keys = []

    def _recv_meta_data(self, keys: List[str]) -> Tuple[int, Dict[str, MetaData]]:
        total_length = int(self.store.get("total_length"))
        meta_data_dict = {}
        for key in keys:
            meta_data_dict[key] = MetaData.from_str(self.store.get(key).decode())
        return total_length, meta_data_dict


    def send_dict(self, tensor_dict: Dict[str, torch.Tensor], dst_rank: int):
        self._send_meta_data(tensor_dict)
        buffer_start = 0

        for key, tensor in tensor_dict.items():
            assert tensor.dtype == self.dtype
            tensor = tensor.flatten()
            length = tensor.numel()


            tensor_start = 0
            while tensor_start < length:
                available_space = self.buffer_length - buffer_start
                tensor_end = min(tensor_start + available_space, length)

                # Copy part of the tensor into the buffer
                self.buffer[buffer_start:buffer_start + (tensor_end - tensor_start)] = tensor[tensor_start:tensor_end]
                buffer_start += tensor_end - tensor_start
                tensor_start = tensor_end

                # If buffer is full, send it
                if buffer_start == self.buffer_length:
                    dist.send(tensor=self.buffer, dst=dst_rank)
                    buffer_start = 0  # Reset buffer

        # Send any remaining data in the buffer
        if buffer_start > 0:
            dist.send(tensor=self.buffer[:buffer_start], dst=dst_rank)

        self._clear_store()


    def recv_dict(self, src_rank: int, keys: List[str]) -> Dict[str, torch.Tensor]:
        # Retrieve metadata from TCPStore
        total_length, meta_data_dict = self._recv_meta_data(keys)
        received_times = math.ceil(total_length / self.buffer_length)
        remaining_length = total_length % self.buffer_length
        assert received_times >= 1
        free_mem = torch.cuda.mem_get_info()[0]/(1024**3)
        # print(f"Before recing dict, free mem: {free_mem:.2f} GB")
        data_tensor = torch.empty(total_length, dtype=self.dtype, device='cuda')
        data_offset = 0

        loop_times = received_times - 1 if remaining_length > 0 else received_times
        for i in range(loop_times):
            dist.recv(self.buffer, src=src_rank)
            data_tensor[data_offset:data_offset + self.buffer_length] = self.buffer
            data_offset += self.buffer_length

        if remaining_length != 0:
            dist.recv(self.buffer[:remaining_length], src=src_rank) 

        data_tensor[data_offset:data_offset + remaining_length] = self.buffer[:remaining_length]

        tensor_dict = {}

        for key, meta_data in meta_data_dict.items():
            offset = meta_data.offset
            length = meta_data.length
            shape = meta_data.shape
            tensor_dict[key] = data_tensor[offset:offset+length].view(shape)

        return tensor_dict
