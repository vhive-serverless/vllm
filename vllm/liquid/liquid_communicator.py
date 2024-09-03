import torch
import torch.distributed as dist
from typing import Dict, Tuple, List
from vllm.liquid.metadata import MetaData
from vllm.liquid.utils import get_tensor_num_bytes
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
        self._prewarm()

    def _send_meta_data(self, tensor_dict: Dict[str, torch.Tensor]):
        offset = 0
        for key, tensor in tensor_dict.items():
            shape = tensor.shape
            length = tensor.numel()
            meta_data = MetaData(offset, length, shape)
            self.store.set(key, str(meta_data))
            self.meta_keys.append(key)
            offset += length
            
        # Send the metadata using TCPStore
        self.store.set(f"total_length", str(offset))
        self.meta_keys.append("total_length")


    def _recv_meta_data(self, keys: List[str]) -> Tuple[int, Dict[str, MetaData]]:
        total_length = int(self.store.get("total_length"))
        meta_data_dict = {}
        for key in keys:
            meta_data_dict[key] = MetaData.from_str(self.store.get(key).decode())
        return total_length, meta_data_dict

    def _clear_store(self):
        for meta_key in self.meta_keys:
            success = self.store.delete_key(meta_key)
            assert success
        self.meta_keys = []

    def _prewarm(self):
        # Only prewarm GPU0 and GPU1
        pre_warm_size = 1000
        if self.rank == 0:
            pre_warm_tensor = torch.randn(pre_warm_size, device='cuda')
            dist.send(pre_warm_tensor, dst=1, group=self.group)
            dist.recv(pre_warm_tensor, src=1, group=self.group)
        elif self.rank == 1:
            pre_warm_tensor = torch.empty(pre_warm_size, device='cuda')
            dist.recv(pre_warm_tensor, src=0, group=self.group)
            dist.send(pre_warm_tensor, dst=0, group=self.group)

    def _send(self, tensor: torch.Tensor, dst: int) -> int:
        # send out the tensor and record the bytes of the tensor
        dist.send(tensor=tensor, dst=dst)
        return get_tensor_num_bytes(tensor)

    def _recv(self, tensor: torch.Tensor, src: int) -> int:
        # recv the tensor and record the bytes of the tensor
        dist.recv(tensor, src=src)
        return get_tensor_num_bytes(tensor)



    def send_dict(self, tensor_dict: Dict[str, torch.Tensor], dst_rank: int) -> int:
        self._send_meta_data(tensor_dict)
        buffer_start = 0

        start = time.time()
        num_bytes_sent = 0
        for key, tensor in tensor_dict.items():
            assert tensor.dtype == self.dtype
            tensor_flatten = tensor.reshape(-1)
            length = tensor_flatten.numel()


            tensor_start = 0
            while tensor_start < length:
                available_space = self.buffer_length - buffer_start
                tensor_end = min(tensor_start + available_space, length)

                # Copy part of the tensor into the buffer
                self.buffer[buffer_start:buffer_start + (tensor_end - tensor_start)].copy_(tensor_flatten[tensor_start:tensor_end])
                buffer_start += tensor_end - tensor_start
                tensor_start = tensor_end

                # If buffer is full, send it
                if buffer_start == self.buffer_length:
                    num_bytes_sent += self._send(tensor=self.buffer, dst=dst_rank)
                    buffer_start = 0  # Reset buffer
            # del tensor_flatten
            # del tensor
            # torch.cuda.empty_cache()
            # free_mem, _ = torch.cuda.mem_get_info()
            # print(f"free_mem after sending tensor: {key}: {free_mem/(1024**2):.2f}MB")

        # Send any remaining data in the buffer
        if buffer_start > 0:
            num_bytes_sent+=self._send(tensor=self.buffer[:buffer_start], dst=dst_rank)
        torch.cuda.synchronize()
        send_latency = time.time() - start

        self._clear_store()
        return num_bytes_sent


    def recv_dict(self, src_rank: int, keys: List[str]) -> Dict[str, torch.Tensor]:
        # Retrieve metadata from TCPStore
        total_length, meta_data_dict = self._recv_meta_data(keys)
        received_times = math.ceil(total_length / self.buffer_length)
        remaining_length = total_length % self.buffer_length
        if remaining_length == 0:
            buffer_lengths = [self.buffer_length for _ in range(received_times)]
        else:
            buffer_lengths = [self.buffer_length for _ in range(received_times - 1)]
            buffer_lengths.append(remaining_length)
        
        assert received_times >= 1
        # free_mem = torch.cuda.mem_get_info()[0]/(1024**3)
        # print(f"Before recing dict, free mem: {free_mem:.2f} GB")
        # data_tensor = torch.empty(total_length, dtype=self.dtype, device='cuda')
        tensor_dict = {}

        for key, meta_data in meta_data_dict.items():
            try:
                tensor_dict[key] = torch.empty(meta_data.length, dtype=self.dtype, device='cuda')
            except:
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info()
                print(f"After empty cache, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")
                tensor_dict[key] = torch.empty(meta_data.length, dtype=self.dtype, device='cuda')

            # free_mem, _ = torch.cuda.mem_get_info() 
            # print(f"After recving {key}, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")


        num_bytes_recv = 0
        buffer_start = self.buffer_length # Make sure the first unread_buffer_length is 0
        received_count = 0
        for key, meta_data in sorted(meta_data_dict.items(), key=lambda item: item[1].offset):
            length = meta_data.length

            tensor = tensor_dict[key]
            tensor_start = 0
            while tensor_start < length:
                unread_buffer_length = self.buffer_length - buffer_start
                if unread_buffer_length <= 0:
                    
                    num_bytes_recv += self._recv(self.buffer[:buffer_lengths[received_count]], src=src_rank)
                    received_count += 1
                    buffer_start = 0 # set buffer start to 0
                    unread_buffer_length = self.buffer_length

                tensor_end = min(tensor_start+unread_buffer_length, length)

                tensor[tensor_start: tensor_end] = self.buffer[buffer_start: buffer_start+ (tensor_end - tensor_start)]

                buffer_start += (tensor_end - tensor_start)
                tensor_start = tensor_end

        torch.cuda.synchronize()

        for key, meta_data in meta_data_dict.items():
            shape = meta_data.shape
            tensor_dict[key] = tensor_dict[key].view(shape)

        return tensor_dict
