
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
        liquid_gpu_range = [0,1,2,3],
        liquid_gpu_space = 32,
        liquid_driver_gpu_id = 0, 
        liquid_total_num_shards = 4,
        # gpu_memory_utilization=0.7,
    )


    torch.cuda.empty_cache()
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"After initializing model, allocated space on GPU 0: {torch.cuda.memory_allocated()/(1024**3):.2f} GB, reserved space on GPU 0: {torch.cuda.memory_reserved()/(1024**3):.2f} GB, free space: {free_mem/(1024**3):.2f}GB")

    for i in range(1):
        liquid_request = LiquidRequest(LiquidType.LIQUID_1_2)
        llm.do_liquid(liquid_request)
        liquid_request = LiquidRequest(LiquidType.LIQUID_2_1)
        llm.do_liquid(liquid_request)

    
    llm.llm_engine.check_liquid_request_and_complete()
    llm.llm_engine.check_liquid_request_and_complete()

if __name__ == '__main__':
    main()