from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")
prompts = [
    "Welcome to LLM system research direction!",
]
sampling_params = SamplingParams(temperature=0, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    #print(output.outputs)
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
