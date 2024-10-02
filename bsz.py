
from vllm import LLM, SamplingParams
# from vllm import EngineArgs, LLMEngine
import asyncio
import torch

import os
import time
import json

# model = "meta-llama/Meta-Llama-3-8B"
model = "facebook/opt-6.7b"
# model_path = os.path.join("./models", model)
device_name = "fuji2"
tp_level = 4

def main():
    llm = LLM(
        model, 
        enforce_eager=True,
        # load_format="auto",
        tensor_parallel_size=tp_level,
        # liquid_gpu_range = [0,1,2,3],
        # liquid_gpu_range = [0,1,2,3],
        # liquid_gpu_space = 32,
        # liquid_driver_gpu_id = 0, 
        # liquid_total_num_shards = 4,
        distributed_executor_backend='mp'

    )
    generated_token_num = 32
    sampling_params = SamplingParams(temperature=0, min_tokens=generated_token_num, max_tokens=generated_token_num)
    bsz = 1
    iteration = 10
    word = "what is LLM?" 
    prompt = word 
    bszs = []
    latencys = []
    tgts = []
    for i in range(iteration):
        inputs = [prompt for _ in range(bsz)]
        start = time.time()
        output = llm.generate(inputs, sampling_params=sampling_params)
        generate_latency = time.time() - start
        tgt = (bsz * generated_token_num) / (generate_latency)
        latencys.append(generate_latency)
        tgts.append(tgt)
        bszs.append(bsz)
        bsz *= 2

    model_suffix = model.split('/')[-1]
    with open(f"bsz_tgt_{model_suffix}_seq{generated_token_num}_{device_name}_tp{tp_level}.json", 'w+') as f:
        json.dump({
            "bszs": bszs,
            "latencys": latencys,
            "tgts": tgts,
        }, f)



        

if __name__ == '__main__':
    # torch.cuda.memory._record_memory_history(context="all", stacks="all")
    main()