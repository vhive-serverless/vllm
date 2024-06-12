from vllm import LLM, SamplingParams
from vllm.core.place import PlaceRequest
import time
prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, top_p=0.95)
start = time.time()
model = "facebook/opt-125m"
llm = LLM(model=model,  tensor_parallel_size=2, enforce_eager=True)
vllm_init_latency = time.time() - start
print(f"vllm initialization latency: {vllm_init_latency:.1f}s")
# outputs = llm.generate(prompts, sampling_params)

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# llm.add_place_request(PlaceRequest())
# outputs = llm.generate(prompts, sampling_params)

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
