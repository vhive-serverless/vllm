from vllm.coldstart_profiler import ColdstartProfiler, DummyProfiler, ENVSTR
from multiprocessing import Manager
import os



MANAGER = Manager()
PROFILER_MAP = MANAGER.dict()

def create_profiler(name: str = "root") -> ColdstartProfiler:
    val = os.getenv(ENVSTR)
    if val is not None:
        profiler = ColdstartProfiler(name=name, manager=MANAGER)
    else:
        profiler = DummyProfiler(name=name, manager=MANAGER)
    PROFILER_MAP[name] = profiler
    return profiler

def fork_profiler(name: str, parent_name: str="root") -> ColdstartProfiler:
    val = os.getenv(ENVSTR)
    if val is None:
        return DummyProfiler(name, manager=Manager)
    parent_profiler = PROFILER_MAP[parent_name]
    child_profiler = create_profiler(name)
    parent_profiler.fork(name, child_profiler)
    return child_profiler

def get_profiler(name: str = "root") -> ColdstartProfiler:
    profiler = PROFILER_MAP[name]
    return profiler