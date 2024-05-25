from replica_runner import ReplicaRunner
from multiprocessing import shared_memory
from vllm.engine.arg_utils import EngineSharedMem
import pickle
import time
# Load The model into CPU

runner = ReplicaRunner(model="facebook/opt-6.7b", device=0, gpu_memory_capacity=32, gpu_memory_space=30, load_only=True)

model = runner.get_model()[0]
print(type(model))
data = pickle.dumps(model)
print(len(data))
# save on disk
with open("model.pkl", "wb") as f:
    f.write(data)
