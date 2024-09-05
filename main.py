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
    for i in range(100):
        liquid_request = LiquidRequest(LiquidType.LIQUID_1_2)
        llm.do_liquid(liquid_request)
        # liquid_request = LiquidRequest(LiquidType.LIQUID_2_4)
        # llm.do_liquid(liquid_request)
        # liquid_request = LiquidRequest(LiquidType.LIQUID_4_2)
        # llm.do_liquid(liquid_request)
        liquid_request = LiquidRequest(LiquidType.LIQUID_2_1)
        llm.do_liquid(liquid_request)

    sampling_params = SamplingParams(temperature=0, min_tokens=127, max_tokens=128)
    request_num = 1
    word = "what is LLM?" 
    prompt = word * 200
    inputs = [prompt for _ in range(request_num)]
    for request_id in range(request_num):
        output = llm.generate(inputs, sampling_params=sampling_params)
        print(f"output: {output[0].outputs[0].text}")


        

if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history(context="all", stacks="all")
    main()
    # torch.cuda.memory._dump_snapshot(f"./torch_mem_dump.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)
    # print(f"dumped finished!")