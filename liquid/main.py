
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)

from vllm.worker.worker_base import WorkerWrapperBase
from vllm.engine.arg_utils import EngineArgs
from functools import partial
from typing import Optional, List, Dict, Any
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
import time

model = "facebook/opt-125m"
engine_args = EngineArgs(model=model)
engine_config = engine_args.create_engine_config()

def _get_worker_kwargs(
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
    """Return worker init args for a given rank."""
    if distributed_init_method is None:
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

    return dict(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        lora_config=engine_config.lora_config,
        vision_language_config=engine_config.vision_language_config,
        speculative_config=engine_config.speculative_config,
        is_driver_worker=rank == 0,
    )

def _create_worker(local_rank: int = 0,
                   rank: int = 0,
                   distributed_init_method: Optional[str] = None):

        worker_module_name = "vllm.worker.worker"
        worker_class_name = "Worker"

        wrapper = WorkerWrapperBase(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
        )
        wrapper.init_worker(**_get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

def main() -> None:

    driver_worker = _create_worker(local_rank=0,rank=0)
    # result_handler.start()
    driver_worker.init_device()
    driver_worker.load_model()
    time.sleep(20)
    

if __name__ == '__main__':
    main()