from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from vllm.executor.proxy_manager import ProxyManager
from vllm.entrypoints.openai.protocol import CreateInstanceRequest, DeleteInstanceRequest
logger = init_logger('vllm.entrypoints.liquid')


# This would be your real builder
def build_proxy_manager() -> ProxyManager:
    print("Building Proxy Manager")
    return ProxyManager(world_size=2)


# Server setup
app = FastAPI()
proxy_manager: ProxyManager = None  # type:ignore

@app.on_event("startup")
def startup_event():
    global proxy_manager
    proxy_manager = build_proxy_manager()

@app.post("/create_instance")
def create_instance(req: CreateInstanceRequest):
    proxy_manager.create_instance(req.uuid, req.gpu_ids)
    return {"status": "created", "uuid": req.uuid}

@app.post("/delete_instance")
def delete_instance(req: DeleteInstanceRequest):
    proxy_manager.delete_instance(req.uuid)
    return {"status": "deleted", "uuid": req.uuid}

if __name__ == "__main__":
    uvicorn.run("proxy_server:app", host="0.0.0.0", port=8000, reload=True)