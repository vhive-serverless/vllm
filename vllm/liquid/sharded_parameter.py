import torch
from torch.nn.parameter import Parameter
from typing import List, Optional, Any, Dict


class ShardedParameter(Parameter):
    def __new__(cls, data=None, requires_grad=False, num_shards=1, shard_dim=0, shard_ids=None, **kwargs):
        # Call the __new__ method of the Parameter class
        instance = super(ShardedParameter, cls).__new__(cls, data, requires_grad)
        return instance

    def __init__(self, 
                 data: torch.Tensor,
                 num_shards: int = 1,
                 shard_dim: int = 0,
                 shard_ids : Optional[List[int]] = None,
                 requires_grad: bool = False,
                 ):
        super().__init__()
        self.requires_grad = requires_grad
        if shard_ids is not None:
            assert num_shards == len(shard_ids), f"num_shards:{num_shards} does not equal to the length of shard_ids: {len(shard_ids)}"
        else:
            shard_ids = list(range(num_shards))

        self.data = data
        self.num_shards: int = num_shards
        self.shard_ids: List[int] = shard_ids.copy()
        self.shard_dim: int = shard_dim

        assert len(self.shape) > shard_dim, f"Tensor's shape: {self.shape} has a dimension of {len(self.shape)}, is smaller or equal to shard_dim: {shard_dim}"
        assert self.shape[self.shard_dim] % self.num_shards == 0, f"Tensor's {self.shard_dim} dim with length: {self.shape[self.shard_dim]} is not divisible by number of shards: {self.num_shards}"

        self.shard_size = self.size(self.shard_dim) // self.num_shards
        pass


    def _get_shard(self, tensor: torch.Tensor, shard_id: int, shard_size: Optional[int]=None) -> torch.Tensor:
        if shard_size is None:
            shard_size = self.shard_size
        index = self.shard_ids.index(shard_id)
        start_index = index * shard_size

        shard = tensor.narrow(self.shard_dim, start_index, shard_size)
        return shard

    def _get_shards(self, tensor: torch.Tensor, start_shard_id: int, end_shard_id: int, shard_size: Optional[int]=None) -> torch.Tensor:
        if shard_size is None:
            shard_size = self.shard_size
        index = self.shard_ids.index(start_shard_id)
        start_index = index*shard_size

        shards = tensor.narrow(self.shard_dim, start_index, shard_size*(end_shard_id - start_shard_id))
        return shards
        

    def get_shard(self, shard_id: int, shard_size: Optional[int] = None) -> torch.Tensor:
        if shard_id not in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} not in self.shard_ids")

        shard = self._get_shard(self.data, shard_id, shard_size)
        return shard

    def get_shards(self, start_shard_id: int, end_shard_id: int, shard_size: Optional[int] = None) -> torch.Tensor:
        shards = self._get_shards(self.data, start_shard_id, end_shard_id, shard_size)
        return shards

    def _delete_shards(self, tensor: torch.Tensor, start_shard_id: int, end_shard_id: int, shard_size: Optional[int]=None) -> torch.Tensor:
        index = self.shard_ids.index(start_shard_id)
        if shard_size is None:
            shard_size = self.shard_size
        start_index = index * shard_size
        before_shard = tensor.narrow(self.shard_dim, 0, start_index)
        after_shard = tensor.narrow(self.shard_dim, start_index + shard_size*(end_shard_id - start_shard_id), tensor.size(self.shard_dim) - start_index - shard_size*(end_shard_id - start_shard_id))
        new_data = torch.cat([before_shard, after_shard], dim=self.shard_dim)
        return new_data

    def _delete_shard(self, tensor: torch.Tensor, shard_id: int, shard_size: Optional[int]=None) -> torch.Tensor:
        index = self.shard_ids.index(shard_id)
        if shard_size is None:
            shard_size = self.shard_size
        start_index = index * shard_size
        # Create views of the tensor parts before and after the shard
        before_shard = tensor.narrow(self.shard_dim, 0, start_index)
        after_shard = tensor.narrow(self.shard_dim, start_index + shard_size, tensor.size(self.shard_dim) - start_index - shard_size)

        # Concatenate the views to form a new tensor
        new_data = torch.cat([before_shard, after_shard], dim=self.shard_dim)
        del before_shard, after_shard
        return new_data

    def delete_shards(self, start_shard_id: int, end_shard_id: int, shard_size: Optional[int] = None) -> None:
        new_data = self._delete_shards(self.data, start_shard_id, end_shard_id, shard_size)
        self.data = new_data

        for shard_id in range(start_shard_id, end_shard_id):
            index = self.shard_ids.index(shard_id)
            self.shard_ids.pop(index)



    def delete_shard(self, shard_id: int) -> None:
        if shard_id not in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} not in self.shard_ids: {self.shard_ids}")
        new_data = self._delete_shard(self.data, shard_id)
        self.data = new_data

        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)


    def _is_appendable(self, shard_data: torch.Tensor) -> bool:
        # # Check if the dimension of shard_dim has the same size as the current shard size
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

        # new_data = self._append_shard(self.data, shard_data)
        # self.data = new_data
        self.data = self._append_shard(self.data, shard_data)

        self.shard_ids.append(shard_id) 

    def append_shards(self, start_shard_id:int, end_shard_id: int, shard_data: torch.Tensor) -> None:
        # self.data = self._append_shard(self.data, shard_data)
        if self.shard_ids == []:
            self.data = shard_data.clone()
        else:
            self.extend_and_load_shard(shard_data)
        for shard_id in range(start_shard_id, end_shard_id):
            self.shard_ids.append(shard_id)

    def _in_place_cat(self, new_tensor: torch.Tensor, old_tensor: torch.Tensor, shard: torch.Tensor):
        # In place cat, old tensor at front, new tensor at back
        first_half, last_half = new_tensor.chunk(dim=self.shard_dim, chunks=2)
        assert first_half.shape == old_tensor.shape
        first_half.copy_(old_tensor)
        assert last_half.shape == shard.shape
        last_half.copy_(shard)

    def extend_and_load_shard(self, shard_data: torch.Tensor) -> None:
        # extend current shard by half along self.shard_dim
        shape = list(self.data.shape)
        shape[self.shard_dim] = shape[self.shard_dim] * 2
        new_shape = torch.Size(shape)
        new_data = torch.empty(
            size=new_shape,
            dtype=self.data.dtype,
            device=self.data.device,
        )
        self._in_place_cat(new_data, self.data, shard_data)
        self.data = new_data


class QKVShardedParameter(ShardedParameter):
    def __init__(self, 
                 data: torch.Tensor,
                 num_shards: int = 1,
                 shard_dim: int = 0,
                 shard_ids : Optional[List[int]] = None,
                 requires_grad: bool = False,
                 num_heads_ratio: int = 1,
                 num_kv_heads_ratio: int = 1,
                 ):
        super().__init__(data, num_shards, shard_dim, shard_ids)
        self.requires_grad = requires_grad
        import math
        d = math.gcd(num_heads_ratio, num_kv_heads_ratio)
        num_heads_ratio = num_heads_ratio // d
        num_kv_heads_ratio = num_kv_heads_ratio // d
        self._num_heads_ratio = num_heads_ratio
        self._num_kv_heads_ratio = num_kv_heads_ratio
        # assert self.size(shard_dim) % 3 == 0, f"QKV parameter must have a length divisible by 3 along dim: {shard_dim}"
        # qkv_shard_size = self.size(shard_dim) // 3
        # self.q_data = self.narrow(shard_dim, 0, qkv_shard_size)
        # self.k_data = self.narrow(shard_dim, qkv_shard_size, qkv_shard_size)
        # self.v_data = self.narrow(shard_dim, 2*qkv_shard_size, qkv_shard_size)
        # self.q_data, self.k_data, self.v_data = data.chunk(3, shard_dim)
    
    def customize_chunk(self, data: torch.Tensor) -> torch.Tensor:
        shape = list(data.shape)
        if self.shard_dim >= len(shape):
            raise ValueError(f"shard_dim: {self.shard_dim} is larger than the number of dimensions of the tensor: {len(shape)}")
        siz = shape[self.shard_dim]
        if siz % (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) != 0:
            raise ValueError(f"QKV parameter must have a length divisible by {self._num_heads_ratio + 2 * self._num_kv_heads_ratio} along dim: {self.shard_dim}")
        q_size = siz // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_heads_ratio
        k_size = siz // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        v_size = siz // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        try:
            q_tensor = torch.narrow(data,self.shard_dim, 0, q_size)
            k_tensor = torch.narrow(data,self.shard_dim, q_size, k_size)
            v_tensor = torch.narrow(data,self.shard_dim, q_size + k_size, v_size)
        except Exception as e:
            raise ValueError(f"shape: {data.shape}, dim: {self.shard_dim}, q_size: {q_size}, k_size: {k_size}, v_size: {v_size} ,Error in customizing chunk: {e}")
        return q_tensor, k_tensor, v_tensor


    def get_shard(self, shard_id: int) -> torch.Tensor:
        q_data, k_data, v_data = self.customize_chunk(self.data)
        q_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_heads_ratio
        k_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        v_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        q_shard = self._get_shard(q_data, shard_id, q_shard_size)
        k_shard = self._get_shard(k_data, shard_id, k_shard_size)
        v_shard = self._get_shard(v_data, shard_id, v_shard_size)

        return q_shard, k_shard, v_shard

    def get_shards(self, start_shard_id: int, end_shard_id: int) -> torch.Tensor:
        q_data, k_data, v_data = self.customize_chunk(self.data)
        q_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_heads_ratio
        k_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        v_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        q_shards = self._get_shards(q_data, start_shard_id, end_shard_id, q_shard_size)
        k_shards = self._get_shards(k_data, start_shard_id, end_shard_id, k_shard_size)
        v_shards = self._get_shards(v_data, start_shard_id, end_shard_id, v_shard_size)
        return q_shards, k_shards, v_shards

    def delete_shard(self, shard_id: int) -> None:
        q_data, k_data, v_data = self.customize_chunk(self.data)
        q_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_heads_ratio
        k_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        v_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        q_data = self._delete_shard(q_data, shard_id, q_shard_size)
        k_data = self._delete_shard(k_data, shard_id, k_shard_size)
        v_data = self._delete_shard(v_data, shard_id, v_shard_size)

        new_data = torch.cat([q_data, k_data, v_data], dim=self.shard_dim)
        self.data = new_data
        # del q_data, k_data, v_data
        # self.q_data, self.k_data, self.v_data = self.data.chunk(3, self.shard_dim)

        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)

    def delete_shards(self, start_shard_id: int, end_shard_id: int):
        q_data, k_data, v_data = self.customize_chunk(self.data)
        q_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_heads_ratio
        k_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        v_shard_size = self.shard_size // (self._num_heads_ratio + 2 * self._num_kv_heads_ratio) * self._num_kv_heads_ratio
        q_data = self._delete_shards(q_data, start_shard_id, end_shard_id, q_shard_size)
        k_data = self._delete_shards(k_data, start_shard_id, end_shard_id, k_shard_size)
        v_data = self._delete_shards(v_data, start_shard_id, end_shard_id, v_shard_size)
        new_data = torch.cat([q_data, k_data, v_data], dim=self.shard_dim)
        self.data = new_data
        # del q_data, k_data, v_data
        # self.q_data, self.k_data, self.v_data = self.data.chunk(3, self.shard_dim)

        for shard_id in range(start_shard_id, end_shard_id):
            index = self.shard_ids.index(shard_id)
            self.shard_ids.pop(index)
        
    
    
    def append_shard(self, shard_id: int, q_shard: torch.Tensor, k_shard: torch.Tensor, v_shard: torch.Tensor) -> None:
        if shard_id in self.shard_ids:
            raise ValueError(f"shard_id: {shard_id} is already in self.shard_ids")


        q_data, k_data, v_data = self.customize_chunk(self.data)
        q_data = self._append_shard(q_data, q_shard)
        k_data = self._append_shard(k_data, k_shard)
        v_data = self._append_shard(v_data, v_shard)

        self.data = torch.cat([q_data, k_data, v_data], dim=self.shard_dim)
        # del self.q_data, self.k_data, self.v_data
        # self.q_data, self.k_data, self.v_data = self.data.chunk(3)

        self.shard_ids.append(shard_id) 

    def append_shards(self, start_shard_id: int, end_shard_id: int,q_shard: torch.Tensor, k_shard: torch.Tensor, v_shard: torch.Tensor) -> None:
        if self.shard_ids == []:
            self.data = torch.cat([q_shard, k_shard, v_shard])
        else:
            self.extend_and_load_shard(q_shard, k_shard, v_shard)
        for shard_id in range(start_shard_id, end_shard_id):
            self.shard_ids.append(shard_id) 


    def extend_and_load_shard(self, q_shard: torch.Tensor, k_shard: torch.Tensor, v_shard: torch.Tensor) -> None:
        # extend current shard by half along self.shard_dim

        shape = list(self.data.shape)
        shape[self.shard_dim] = shape[self.shard_dim] * 2
        new_shape = torch.Size(shape)
        new_data = torch.empty(
            size=new_shape,
            dtype=self.data.dtype,
            device=self.data.device,
        )
        q_data, k_data, v_data = self.customize_chunk(self.data)
        new_q_data, new_k_data, new_v_data = self.customize_chunk(new_data)

        self._in_place_cat(new_q_data, q_data, q_shard)
        self._in_place_cat(new_k_data, k_data, k_shard)
        self._in_place_cat(new_v_data, v_data, v_shard)
        # self.q_data, self.k_data, self.v_data = self.data.chunk(3, dim=self.shard_dim)

        self.data = new_data

# TODO: current is llama3 only
class GateUpShardedParameter(ShardedParameter):
    def __init__(self,
                data: torch.Tensor,
                num_shards: int = 1,
                shard_dim: int = 0,
                shard_ids: Optional[List[int]] = None,
                requires_grad: bool = False,
                ):
        super().__init__(data, num_shards, shard_dim, shard_ids)
        self.requires_grad = requires_grad
        self.shard_size = self.shard_size // 2
        assert self.size(shard_dim) % 2 == 0, f"merged column parameter must have a length divisible by 2 along dim: {shard_dim}"
    
    def get_shards(self, start_shard_id: int, end_shard_id: int) -> torch.Tensor:
        gate_data, up_data = self.data.chunk(2, self.shard_dim)
        gate_shards = self._get_shards(gate_data, start_shard_id, end_shard_id)
        up_shards = self._get_shards(up_data, start_shard_id, end_shard_id)
        return gate_shards, up_shards

    def delete_shards(self, start_shard_id: int, end_shard_id: int) -> None:
        gate_data, up_data = self.data.chunk(2, self.shard_dim)
        gate_data = self._delete_shards(gate_data, start_shard_id, end_shard_id)
        up_data = self._delete_shards(up_data, start_shard_id, end_shard_id)
        new_data = torch.cat([gate_data, up_data], dim=self.shard_dim)
        self.data = new_data
        for shard_id in range(start_shard_id, end_shard_id):
            index = self.shard_ids.index(shard_id)
            self.shard_ids.pop(index)

    def append_shards(self, start_shard_id: int, end_shard_id: int, gate_shard: torch.Tensor, up_shard: torch.Tensor) -> None:
        if self.shard_ids == []:
            self.data = torch.cat([gate_shard, up_shard])
        else:
            self.extend_and_load_shard(gate_shard, up_shard)
        for shard_id in range(start_shard_id, end_shard_id):
            self.shard_ids.append(shard_id)
    
    def extend_and_load_shard(self, gate_shard: torch.Tensor, up_shard: torch.Tensor) -> None:
        shape = list(self.data.shape)
        shape[self.shard_dim] = shape[self.shard_dim] * 2
        new_shape = torch.Size(shape)
        new_data = torch.empty(
            size=new_shape,
            dtype=self.data.dtype,
            device=self.data.device,
        )
        gate_data, up_data = self.data.chunk(chunks=2, dim=self.shard_dim)
        new_gate_data, new_up_data = new_data.chunk(chunks=2, dim=self.shard_dim)

        self._in_place_cat(new_gate_data, gate_data, gate_shard)
        self._in_place_cat(new_up_data, up_data, up_shard)

        self.data = new_data