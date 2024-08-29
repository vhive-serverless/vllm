from fastapi import FastAPI
from server import HttpRequestBody, UvicornServer
import uvicorn
import subprocess
from vllm import LLM, SamplingParams
import threading

model_name = "facebook/opt-6.7b"
class LiquidServer:
    def __init__(self) -> None:
        self.fastapi_app = FastAPI()
        self.llm = LLM(
            model_name, 
            enforce_eager=True,
            # load_format="auto",
            # tensor_parallel_size=2,
            liquid_gpu_range = [0,1],
            liquid_gpu_space = 32,
            liquid_driver_gpu_id = 0, 
            liquid_total_num_shards = 2,
            # gpu_memory_utilization=0.7,
        )
        @self.fastapi_app.post("/v1/completions")
        async def enqueue_request(r: HttpRequestBody) -> None:
            print(f"{r.request_id} received!")
            sampling_params = SamplingParams(max_tokens=r.max_response_length+1, min_tokens=r.max_response_length, temperature=0)
            self.llm._add_request(
                inputs=r.prompt,
                params=sampling_params
            )
        self.http_server = UvicornServer(
            uvicorn.Config(
                app=self.fastapi_app,
                host="localhost",
                port=8000,
            )
        )

    def start(self):

        self.http_server.start()

        command = [
                './LLMLoadgen',
                '-pattern', 'azure-multiplex-70-5',
                '-dataset', 'azure-multiplex',
                '-dst', 'liquid',
                '-ip', 'localhost',
                '-port', '8000',
                '-limit', '100',
                '-max_drift', '100',
                '-model_name', f'{model_name}'
            ]
        working_dir = './LLMLoadgen/LLMLoadgen-0.9/release'
        subprocess.Popen(command, cwd=working_dir)
        try:
            while True:
                self.llm._run_engine(use_tqdm=False)
        except KeyboardInterrupt:
            print(f"vllm engine exit")

if __name__ == '__main__':
    liquid_server = LiquidServer()
    liquid_server.start()
