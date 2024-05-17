import threading
from time import sleep

from fastapi import FastAPI
from uvicorn import Config, Server

from typing import Any
import contextlib
import time

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
