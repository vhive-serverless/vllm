import torch
from torch import Tensor
from typing import List, Optional, Any, Dict

class ShardedTensor(Tensor):
    def __new__(cls, data=None, num_shards=1, shard_dim=0, shard_ids=None, *args, **kwargs):
        # Call the __new__ method of the Parameter class
        instance = super(ShardedTensor, cls).__new__(cls, torch.empty(0), *args, **kwargs)
    
        return instance

    def __init__(self, 
                 data: torch.Tensor,
                 num_shards: int = 1,
                 shard_dim: int = 0,
                 shard_ids : Optional[List[int]] = None
                 ):
        super().__init__()
        self.data = data
        if shard_ids:
            assert num_shards == len(shard_ids), f"num_shards:{num_shards} does not equal to the length of shard_ids: {len(shard_ids)}"
        else:
            shard_ids = list(range(num_shards))

        self.total_num_shards: int = num_shards
        self.shard_ids: List[int] = shard_ids.copy()
        self.shard_dim: int = shard_dim

        assert len(self.shape) > shard_dim, f"Tensor's shape: {self.shape} has a dimension of {len(self.shape)}, is smaller or equal to shard_dim: {shard_dim}"
        assert self.shape[self.shard_dim] % self.total_num_shards == 0, f"Tensor's {self.shard_dim} dim with length: {self.shape[self.shard_dim]} is not divisible by number of shards: {self.total_num_shards}, tensor's shape: {self.shape}"

        self.shard_size = self.shape[self.shard_dim] // self.total_num_shards

    def _copy_new_data(self, data: torch.Tensor):
        shape = data.shape
        self.data = torch.empty(shape)
        self.copy_(data)

    def _get_shard(self, tensor: torch.Tensor, shard_id: int) -> torch.Tensor:
        index = self.shard_ids.index(shard_id)
        start_index = index * self.shard_size

        shard = tensor.narrow(self.shard_dim, start_index, self.shard_size)
        return shard

    def get_shard(self, shard_id: int) -> Tensor:
        if shard_id not in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} not in self.shard_ids")

        shard = self._get_shard(self.data, shard_id)
        return shard

    def _delete_shard(self, tensor: torch.Tensor, shard_id: int) -> torch.Tensor:
        index = self.shard_ids.index(shard_id)

        start_index = index * self.shard_size
        # Create views of the tensor parts before and after the shard
        before_shard = tensor.narrow(self.shard_dim, 0, start_index)
        after_shard = tensor.narrow(self.shard_dim, start_index + self.shard_size, self.size(self.shard_dim) - start_index - self.shard_size)

        # Concatenate the views to form a new tensor
        new_data = torch.cat([before_shard, after_shard], dim=self.shard_dim)
        return new_data

    def delete_shard(self, shard_id: int) -> None:
        if shard_id not in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} not in self.shard_ids")
        new_data = self._delete_shard(self.data, shard_id)
        self.data = new_data

        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)


    def _is_appendable(self, shard_data: torch.Tensor) -> bool:
        # Check if the dimension of shard_dim has the same size as the current shard size
        # if shard_data.size(self.shard_dim) != self.shard_size:
        #     return False
        
        # Check if all other dimensions are identical to the current data
        for dim in range(shard_data.dim()):
            if dim != self.shard_dim and shard_data.size(dim) != self.size(dim):
                return False
        
        return True

    def _append_shard(self, src_data: torch.Tensor, appended_data: torch.Tensor) -> torch.Tensor:

        new_data = torch.cat([src_data, appended_data], dim=self.shard_dim)
        return new_data

    def append_shard(self, shard_id: int, shard_data: torch.Tensor) -> None:
        if shard_id in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} is already in self.shard_ids")

        if not self._is_appendable(shard_data):
            raise ValueError(f"data with shape: {shard_data.shape} cannot be appended to tensor with shape: {self.shape}")

        new_data = self._append_shard(self.data, shard_data)
        self.data = new_data

        self.shard_ids.append(shard_id) 

        


class QKVShardedTensor(ShardedTensor):
    def __init__(self, 
                 data: torch.Tensor,
                 num_shards: int = 1,
                 shard_dim: int = 0,
                 shard_ids : Optional[List[int]] = None
                 ):
        super().__init__(data, num_shards, shard_dim, shard_ids)
        assert self.size(shard_dim) % 3 == 0, f"QKV parameter must have a length divisible by 3 along dim: {shard_dim}"
        qkv_shard_size = self.size(shard_dim) // 3
        self.q_data = self.narrow(shard_dim, 0, qkv_shard_size)
        self.k_data = self.narrow(shard_dim, qkv_shard_size, qkv_shard_size)
        self.v_data = self.narrow(shard_dim, 2*qkv_shard_size, qkv_shard_size)

    def get_shard(self, shard_id: int) -> torch.Tensor:
        q_shard = self._get_shard(self.q_data, shard_id)
        k_shard = self._get_shard(self.k_data, shard_id)
        v_shard = self._get_shard(self.v_data, shard_id)

        shard = torch.cat([q_shard, k_shard, v_shard], dim=self.shard_dim)
        return shard

    def delete_shard(self, shard_id: int) -> None:
        new_q_data = self._delete_shard(self.q_data, shard_id)
        new_k_data = self._delete_shard(self.k_data, shard_id)
        new_v_data = self._delete_shard(self.v_data, shard_id)

        new_data = torch.cat([new_q_data, new_k_data, new_v_data], dim=self.shard_dim)
        self.data = new_data

        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)
    
    
    def append_shard(self, shard_id: int, shard_data: torch.Tensor) -> None:
        if shard_id in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} is already in self.shard_ids")

        if not self._is_appendable(shard_data):
            raise ValueError(f"data with shape: {shard_data.shape} cannot be appended to tensor with shape: {self.shape}")

        qkv_shard_size = shard_data.size(self.shard_dim) // 3
        appended_q_data = shard_data.narrow(self.shard_dim, 0, qkv_shard_size)
        appended_k_data = shard_data.narrow(self.shard_dim, qkv_shard_size, qkv_shard_size)
        appended_v_data = shard_data.narrow(self.shard_dim, 2*qkv_shard_size, qkv_shard_size)

        new_q_data = self._append_shard(self.q_data, appended_q_data)
        new_k_data = self._append_shard(self.k_data, appended_k_data)
        new_v_data = self._append_shard(self.v_data, appended_v_data)

        self.data = torch.cat([new_q_data, new_k_data, new_v_data], dim=self.shard_dim)

        self.shard_ids.append(shard_id) 