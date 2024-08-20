import torch
import threading
import time

class CudaMemoryProfiler:
    def __init__(self, interval: float = 1.0,cuda_index: int = 0):
        self.interval = interval  # Interval between memory recordings in seconds
        self.cuda_index = cuda_index
        self.memory_records = []
        self._running = False
        self._thread = None

    def _record_memory(self):
        while self._running:
            # Get current GPU memory allocated in GB
            memory_gb = torch.cuda.memory_allocated(self.cuda_index) / (1024**3)
            self.memory_records.append(memory_gb)
            time.sleep(self.interval)

    def __enter__(self):
        # Start the recording thread
        self._running = True
        self._thread = threading.Thread(target=self._record_memory)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the recording thread
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def get_memory_records(self):
        return self.memory_records
