import time
import subprocess
import json
from collections import OrderedDict
from typing import Dict

ENVSTR = "COLDSTART_REPORT_FILEPATH"
# when disabled, simply do nothing
class DummyProfiler:
    def __init__(self, name="root", parent=None, parent_event_name=None, manager=None):
        pass
    # ----------- Core Recording Methods -----------
    def mark(self, name: str):
        pass

    def fork(self, child_name: str, child_profiler):
        pass

    def end_fork(self, child):
        pass

    def report(self) -> str:
        """Return the formatted report for this profiler and all children."""
        return f"{ENVSTR} is not set!"

    # ----------- JSON Dump -----------

    def dump_json(self):
        """Dump full profiler data (hierarchy + events + GPU) to JSON."""
        pass
        

class ColdstartProfiler:
    def __init__(self, name="root", parent=None, parent_event_name=None, manager=None):
        self.name = name
        self.parent = parent
        self.parent_event_name = parent_event_name

        if manager is None:
            raise ValueError("ColdstartProfiler requires a manager instance")

        self.events = manager.dict()
        self.children = manager.list()

    # ----------- GPU Memory Snapshot -----------
    def _get_gpu_memory(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            gpu_usage = {}
            for line in output.strip().split("\n"):
                idx, used, total = [x.strip() for x in line.split(",")]
                gpu_usage[int(idx)] = {
                    "used_MB": float(used),
                    "total_MB": float(total),
                }
            return gpu_usage
        except Exception as e:
            return {"error": str(e)}

    # ----------- Core Recording Methods -----------
    def mark(self, name: str):
        """Record a named event with timestamp and GPU usage."""
        t = time.time()
        gpu_usage = self._get_gpu_memory()
        self.events[name] = {"time": t, "gpu_usage": gpu_usage}

    def fork(self, child_name: str, child_profiler):
        """Fork a new child time axis. The child starts from the current time."""
        self.children.append(child_profiler)
        # mark fork start in parent
        self.mark(f"fork_{child_name}_begin")
        # initialize child's first event
        child_profiler.mark("start")

    def end_fork(self, child):
        """Mark the end of a child axis in the parent timeline."""
        self.mark(f"fork_{child.name}_end")

    # ----------- Reporting -----------
    def _generate_report(self, indent=0) -> str:
        prefix = "  " * indent
        lines = []

        names = list(self.events.keys())
        times = [self.events[n]["time"] for n in names] if self.events else []

        if len(times) >= 2:
            total_time = times[-1] - times[0]
            lines.append(f"{prefix}----{{{self.name}: e2e_latency:{total_time:.3f} second}}----")
            for i in range(1, len(names)):
                delta = times[i] - times[i - 1]
                perc = (delta / total_time) * 100
                lines.append(f"({perc:.1f}%){prefix}{names[i-1]} → {names[i]}: {delta:.3f} sec")
        else:
            lines.append(f"{prefix}----{{{self.name}: insufficient events}}----")

        # Recursively include child summaries
        for child in self.children:
            lines.append("")  # spacing
            lines.append(child._generate_report(indent=indent + 1))

        return "\n".join(lines)

    def report(self) -> str:
        """Return the formatted report for this profiler and all children."""
        return self._generate_report()

    # ----------- JSON Dump -----------
    def _serialize(self):
        data = {
            "name": self.name,
            "events": self.events,
            "children": [child._serialize() for child in self.children]
        }
        return data

    def dump_json(self):
        filepath = os.getenv(ENVSTR)
        """Dump full profiler data (hierarchy + events + GPU) to JSON."""
        with open(filepath, "w") as f:
            json.dump(self._serialize(), f, indent=2)
        print(f"[INFO] Profiling data dumped to {filepath}")
