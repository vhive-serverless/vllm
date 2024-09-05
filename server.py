from pydantic import BaseModel
import threading
from time import sleep

from fastapi import FastAPI
from uvicorn import Config, Server

from typing import Any
import contextlib
import time


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
    global_request_id: int
    prompt_length: int


class UvicornServer(Server):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def install_signal_handlers(self) -> None:
        pass


    def start(self) -> None:
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self) -> None:
        self.should_exit = True
        if self.thread:
            self.thread.join()