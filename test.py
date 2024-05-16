from vllm import LLM, SamplingParams
import time
prompts = [
    "Reply with repeated a"*100,
    "The future of the"*100,
    "The future of what is saldkf;lsdfk"*100,
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm_0 = LLM(model="facebook/opt-125m", local_rank=0, enforce_eager=True)
outputs_0 = llm_0.generate(prompts, sampling_params)


# Print the outputs.
print(f"----------------Results for outputs_0:----------------")
for output in outputs_0:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

