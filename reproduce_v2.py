from vllm import LLM, SamplingParams
from vllm.liquid.request import LiquidRequest, LiquidType
# from vllm import EngineArgs, LLMEngine
import asyncio
import torch

import os

# model = "meta-llama/Meta-Llama-3-8B"
model = "facebook/opt-6.7b"
# model_path = os.path.join("./models", model)

def main():
    llm = LLM(
        model, 
        enforce_eager=True,
        # load_format="auto",
        # tensor_parallel_size=2,
        liquid_gpu_range = [0,1],
        liquid_gpu_space = 32,
        liquid_driver_gpu_id = 0, 
        liquid_total_num_shards = 2,
        # gpu_memory_utilization=0.7,
    )


    torch.cuda.empty_cache()
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"After initializing model, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")

    shards_weights = llm.llm_engine.model_executor.get_shards_weights(shard_ids=[1])
    cpu_shards_weights = {}
    for name, tensor in shards_weights.items():
        cpu_shards_weights[name] = tensor.to("cpu")
        # del tensor

    # for name, tensor in shards_weights.items():
    #     del tensor
    del shards_weights

    
    torch.cuda.empty_cache()
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"After sending weights to cpu and deleting original weights, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")

    llm.llm_engine.model_executor.delete_shards_weights(shard_ids=[1])
    received_shards_weights = {}
    for name, _ in cpu_shards_weights.items():
        received_shards_weights[name] = cpu_shards_weights[name].to("cuda")

    torch.cuda.empty_cache()
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"After recving weights from cpu, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")
    llm.llm_engine.model_executor.append_shards_weights(shard_ids=[1], shards_weights=received_shards_weights)

    torch.cuda.empty_cache()
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"After appending, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")





if __name__ == '__main__':
    main()