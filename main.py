from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
from vllm import EngineArgs, LLMEngine
import asyncio

import os

model = "facebook/opt-125m"
# model_path = os.path.join("./models", model)

def main():
    llm = LLM(
        model, 
        enforce_eager=True,
        load_format="auto",
        # tensor_parallel_size=2,
        liquid_gpu_range = [0,1],
        liquid_gpu_space = 32,
        liquid_driver_gpu_id = 0, 
        liquid_total_num_shards = 4,
        gpu_memory_utilization=0.3
    )
    shard_ids = [3]
    src = 0
    dst = 1
    llm.do_liquid(shard_ids, src, dst)
    # llm.do_liquid(shard_ids, dst, src)
    # llm.do_liquid(shard_ids, src, dst)

    sampling_params = SamplingParams(temperature=0)
    request_num = 10
    for request_id in range(request_num):
        output = llm.generate(f"What is LLM?", sampling_params=sampling_params)
        print(output[0].outputs[0].text)


        

if __name__ == '__main__':
    main()