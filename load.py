from replica_runner import ReplicaRunner
from multiprocessing import shared_memory
from vllm.engine.arg_utils import EngineSharedMem
import pickle
import time
# Load The model into CPU
# save on disk
# read model.pkl as data
with open("model.pkl", "rb") as f:
    data = f.read()
shm = shared_memory.SharedMemory(create=True, size = len(data))
shm.buf[:len(data)] = data

engine_conf = EngineSharedMem(shm.name, len(data))
print("Done")
time.sleep(5)
t = time.time()
real_runner = ReplicaRunner(model="facebook/opt-6.7b", device=0, gpu_memory_capacity=32, gpu_memory_space=30, load_position=engine_conf)
duration = time.time() - t
print(duration)
shm.close()
shm.unlink()
