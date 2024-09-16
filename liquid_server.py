from fastapi import FastAPI
from server import HttpRequestBody, UvicornServer
import uvicorn
import subprocess
from vllm import LLM, SamplingParams, RequestOutput
from typing import List, Dict, Tuple
import json
import traceback
from vllm.liquid.request import LiquidRequest, LiquidType

model_name = "facebook/opt-6.7b"
output_file_name = "./output.txt"
class LiquidServer:
    def __init__(self) -> None:
        self.fastapi_app = FastAPI()
        self.llm = LLM(
            model_name, 
            enforce_eager=True,
            # load_format="auto",
            # tensor_parallel_size=2,
            liquid_gpu_range = [0,1,2,3],
            liquid_gpu_space = 32,
            liquid_driver_gpu_id = 0, 
            liquid_total_num_shards = 4,
            gpu_memory_utilization=0.85,
        )
        with open(output_file_name, "w"):
            pass

        self.f = open(output_file_name, "a")
        self.llm.llm_engine.request_output_handler = self.f 
        @self.fastapi_app.post("/v1/completions")
        async def enqueue_request(r: HttpRequestBody) -> None:
            print(f"{r.request_id} received!")
            max_model_length = self.llm.llm_engine.model_config.max_model_len
            if r.max_response_length + r.prompt_length > max_model_length:
                return
            sampling_params = SamplingParams(max_tokens=r.max_response_length+1, min_tokens=r.max_response_length, temperature=0)
            self.llm._add_request(
                inputs=r.prompt,
                global_id=r.global_request_id,
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
                '-pattern', 'azure-code-130-5',
                '-dataset', 'azure-code',
                '-dst', 'liquid',
                '-ip', 'localhost',
                '-port', '8000',
                '-max_drift', '100',
                '-model_name', f'{model_name}'
            ]
        working_dir = '/home/lrq/baseline/LLMLoadgen/release'
        loadgen_process = subprocess.Popen(command, cwd=working_dir)
        loadgen_running = True
        request_outputs: List[RequestOutput] = []
        print("Loadgen started!")
        try:
            while loadgen_running:
                self.llm._run_engine(use_tqdm=False)
                while self.llm.llm_engine.request_output_queue.qsize() != 0:
                    request_output = self.llm.llm_engine.request_output_queue.get()
                    # print(f"request: {request_output.request_id} finished!")
                    request_outputs.append(request_output)

                loadgen_running = (loadgen_process.poll() is None)
        except Exception as e:
            stack_trace = traceback.format_exc()
            print(f"Error: {e}, stack trace: {stack_trace}")
        finally:
            loadgen_process.terminate()
            print(f"loadgen process terminated!")

            # store all the results
            timestamps = self.llm.llm_engine.auto_scaler.timestamp_records
            tp_level_records = self.llm.llm_engine.auto_scaler.tp_level_records
            cache_usage_records = self.llm.llm_engine.auto_scaler.cache_usage_records
        
            arrival_times = []
            e2e_latencys = []
            queueing_latencys = []
            serving_latencys = []
            for request_output in request_outputs:
                metrics = request_output.metrics
            
                e2e_latency = metrics.finished_time - metrics.arrival_time
                queueing_latency = metrics.time_in_queue if metrics.time_in_queue else 0
                serving_latency = e2e_latency - queueing_latency

                arrival_times.append(metrics.arrival_time)
                e2e_latencys.append(e2e_latency)
                queueing_latencys.append(queueing_latency)
                serving_latencys.append(serving_latency)

            data = {
                "timestamps": timestamps,
                "tp_level_records": tp_level_records,
                "cache_usage_records": cache_usage_records,
                "arrival_times": arrival_times,
                "e2e_latencys": e2e_latencys,
                "queueing_latencys": queueing_latencys,
                "serving_latencys": serving_latencys,
            }

            # Dump the data to a JSON file
            with open('liquid_results.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)  # indent=4 for pretty printing
        
        self.f.close()
        del self.llm
        self.http_server.stop()


if __name__ == '__main__':
    liquid_server = LiquidServer()
    liquid_server.start()
