from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn.config
from uvicorn_server import UvicornServer
from attr import dataclass
from queue import Queue
from threading import Thread, Event
import time
# from replica_runner import ReplicaRunner
from liquid_runner import LiquidRunner
from vllm.core.place import PlaceRequest
import uvicorn
import subprocess
from vllm.logger import init_logger
from typing import Tuple, List, Dict

logger = init_logger(__name__)
class HttpRequestBody(BaseModel):
    """Pydantic class storing contents of http request body

    Attr:
        request_id : Id for request
        prompt: Input Prompt for request
        max_response_length: Maximum response length
        model_name: Which model this request is targeted
    """
    model: str
    prompt: str
    request_id: int
    max_response_length: int

@dataclass
class Request:
    model: str = ""
    prompt: str = ""
    request_id: int = -1
    max_response_length: int = -1

    def from_http_request_body(self, body: HttpRequestBody):
        self.model = body.model
        self.prompt = body.prompt
        self.request_id = body.request_id
        self.max_response_length = body.max_response_length

class LLMServer:
    def __init__(self) -> None:
        self.app = FastAPI()
        self.request_queue:Queue[Request] = Queue()
        self.http_server = None
        self.process_thread = None
        self.runner: LiquidRunner = None
        self.stop_flag: Event = Event()
        self.num_refused_requests = 0

    def init(self) -> None:
        
        self.runner = LiquidRunner(model="facebook/opt-6.7b", device=0, gpu_memory_capacity=32, gpu_memory_space=30)
        @self.app.post("/v1/completions")
        async def enqueue_request(r : HttpRequestBody) -> None:
            request : Request = Request()
            request.from_http_request_body(r)
            logger.info(f"Request {request.request_id} received")
            self.request_queue.put(request)
        
        uvicorn_config = uvicorn.Config(app=self.app, host="localhost", port=8000)
        self.http_server = UvicornServer(config=uvicorn_config)
        self.process_thread = Thread(target=self.process)

    def start(self) -> None:
        if self.http_server:
            self.http_server.start()

        if self.process_thread:
            self.process_thread.start()

        if self.runner:
            self.runner.start()

    def stop(self) -> None:
        if self.http_server:
            self.http_server.stop()
        
        self.stop_flag.set()

    def place(self, layer_range: Tuple[int,int], dest_gpu_id: int) -> None:
        place_request: PlaceRequest = PlaceRequest(
            layer_range=layer_range,
            dest_gpu_id=dest_gpu_id
        )
        self.runner.add_place_request(place_request=place_request)

    def process(self) -> None:
        while not self.stop_flag.is_set():
            if self.request_queue.empty():
                time.sleep(1)
                continue
            request = self.request_queue.get()
            prompt = request.prompt
            token_length = request.max_response_length
            request_id = request.request_id
            
            ret = self.runner.serve(prompts=prompt, token_length=token_length, global_request_id=request_id)
            if ret == False:
                self.num_refused_requests += 1

llm_server = LLMServer()
llm_server.init()
llm_server.start()

def place(timeout: int, times=1) -> None:
    for i in range(times):
        if i % 4 == 0:
            llm_server.place((0,16), 1)
        elif i % 4 == 1:
            llm_server.place((0,16), 0)
        elif i % 4 == 2:
            llm_server.place((0,6), 0)
        elif i % 4 == 3:
            llm_server.place((6,12), 0)
        time.sleep(timeout)

place_thread = Thread(target=place, args=(10,1))
place_thread.start()


request_limit = 128
command = [
    './LLMLoadgen',
    '-pattern', 'azure-conv-100-10',
    '-sim_dir', './simulator',
    '-dataset', 'azure',
    '-dst', 'liquid',
    '-ip', 'localhost',
    '-port', '8000',
    '-limit', f'{request_limit}',
    '-max_drift', '100'
]
working_dir = './LLMLoadgen/release'
results = subprocess.run(command, cwd=working_dir, capture_output=True, text=True)
if results.returncode == 0:
    print(f"Loadgen exit successfully!")
else:
    print(f"Loadgen exit with error: {results.stderr}")

while len(llm_server.runner.finished_requests) + llm_server.num_refused_requests < request_limit:
    time.sleep(1)
llm_server.stop()
print(f"Number of refused reqeusts: {llm_server.num_refused_requests}/{request_limit}")
