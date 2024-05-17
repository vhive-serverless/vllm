"""Replica Runner"""
from typing import List, Dict
from queue import Queue
from threading import Thread

from vllm import LLM, SamplingParams
# from protos import ServeRequest
from vllm.engine.metrics import EngineMetrics
from vllm.outputs import RequestOutput

import torch

from vllm.logger import init_logger
logger = init_logger(__name__)

class ReplicaRunner(LLM):  # type: ignore
    def __init__(self, model: str, device: int, gpu_memory_capacity:int, gpu_memory_space:int):
        gpu_memory_utilization: float = gpu_memory_space / gpu_memory_capacity
        logger.debug(f"Replica runner inited with gpu utilization: {gpu_memory_utilization:.2f}")
        super().__init__(model, local_rank=device, enforce_eager=True, gpu_memory_utilization=gpu_memory_utilization, device=f'cuda:{device}')

        self.request_map: Dict[str, int] = {}
        # self.running_status: bool = True
        self.running_thread = Thread(target=self._run)
        self.running_thread.daemon = True
        # self.running_thread.start()
        self.finished_requests: List[int] = []

    def start(self) -> None:
        self.running_status: bool = True
        self.running_thread.start()

    def stop(self) -> None:
        self.running_status = False
        logger.info("Stop ReplicaRunner")
        self.running_thread.join()
        logger.info("ReplicaRunner stopped")

    def serve(self, prompts: str, token_length: int, global_request_id: int) -> bool:
        """Serve the request
        """
        if not self.running_status:
            logger.error("ReplicaRunner is stopped")
            return False
        logger.info(f"Serve request: {prompts} with request_id: {global_request_id}")
        samplingMethod = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=token_length)
        return self._add_request_with_check(prompts, samplingMethod, global_request_id)

    def metrics(self) -> EngineMetrics:
        """Get the metrics
        """
        return self.llm_engine.get_latest_metrics()

    def all_metrics(self) -> List[EngineMetrics]:
        """Get all history metrics
        """
        metrics = self.llm_engine.get_metrics_history()
        return metrics  # type: ignore

    def _add_request_with_check(self, prompt: str, sampling_params: SamplingParams, global_request_id: int) -> bool:
        metrics = self.metrics()
        if metrics.gpu_cache_usage > 0.9 and (metrics.num_waiting > 0 or metrics.num_swapped > 0):
            # refuse as there's waiting and swapped already
            logger.error(
                f"Refuse request as there's waiting and swapped already")
            return False
        internal_request_id = str(next(self.request_counter))
        self.llm_engine.add_request(internal_request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids=None,
                                    lora_request=None,
                                    prefix_pos=None)
        self.request_map[internal_request_id] = global_request_id
        return True

    def _process_output(self, output: RequestOutput) -> None:
        if output.request_id not in self.request_map:
            # Unknown request id
            logger.error(f"Unknown request id: {output.request_id}")
            return
        global_request_id = self.request_map.pop(output.request_id)
        logger.info(f"Request id: {global_request_id} finished")
        self.finished_requests.append(global_request_id)

    def _engine_run(self) -> None:
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    self._process_output(output)

    def _run(self) -> None:
        while self.running_status:
            self._engine_run()
        # running status stopped, only process the remaining requests
        self._engine_run()
