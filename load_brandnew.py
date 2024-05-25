from replica_runner import ReplicaRunner
import time
t = time.time()
real_runner = ReplicaRunner(model="facebook/opt-6.7b", device=0, gpu_memory_capacity=32, gpu_memory_space=30)
duration = time.time() - t
print(duration)
