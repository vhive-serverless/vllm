
from vllm import LLM, SamplingParams
from vllm.liquid.request import LiquidRequest, LiquidType
# from vllm import EngineArgs, LLMEngine
import asyncio
import torch

import os
import random
import time
import json
from vllm.inputs import TokensPrompt

model = "meta-llama/Meta-Llama-3-8B"
# model = "facebook/opt-6.7b"
# model_path = os.path.join("./models", model)

def main():
    tp_level = 4
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
        distributed_executor_backend='mp',

    )
    sampling_params = SamplingParams(temperature=0, min_tokens=1, max_tokens=1)
    request_num = 1
    word = "what is LLM?" 
    prompt = word 
    min_value = 100
    max_value = 100
    seq_lens = range(10, 8192, 32)
    latencys = []
    for seq_len in seq_lens:
        inputs = TokensPrompt(prompt_token_ids=[random.randint(min_value, max_value) for _ in range(seq_len)])
        # inputs.prompt_token_ids = [0 for _ in range(seq_len)]
        start = time.time()
        
        output = llm.generate(prompts=inputs, sampling_params=sampling_params)
        # llm.generate(f"What is LLM", sampling_params=sampling_params)
        latency = time.time() - start
        print(f"It takes {latency:.2f}s to prefill {seq_len} tokens")
        latencys.append(latency)

    with open(f"prefill_latency_tp{tp_level}.json", "w") as f:
        json.dump({
            "seq_lens": list(seq_lens),
            "latencys": latencys,
        }, f)






        

if __name__ == '__main__':
    main()