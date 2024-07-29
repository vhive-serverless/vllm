from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
import asyncio

import os

model = "facebook/opt-125m"
model_path = os.path.join("./models", model)

async def main():
    engine_args = AsyncEngineArgs(
        model_path, 
        enforce_eager=True,
        load_format="serverless_llm",
        liquid_gpu_range = [0,1],
        liquid_gpu_space = 32,
        liquid_driver_gpu_id = 0, 
        liquid_total_num_shards = 4,
    )
    async_engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args)

    async_engine.start_background_loop()

    sampling_params = SamplingParams(temperature=0)
    results_generators = []
    request_num = 10
    for request_id in range(request_num):
        results_generators.append(async_engine.generate(f"What is LLM?", sampling_params=sampling_params, request_id=f"{request_id}"))

    # shard_ids = [2,3]
    # src = 0
    # dst = 1
    # async_engine.do_liquid(shard_ids, src, dst)

    for results_generator in results_generators:
        final_output = None
        async for output in results_generator:
            final_output = output

        print(f"output for request_id: [{final_output.request_id}]: {final_output.outputs[0].text}")
        

if __name__ == '__main__':
    asyncio.run(main())