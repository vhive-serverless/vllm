#!/bin/bash

pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

# 遍历所有 PID 并杀掉相应的进程
for pid in $pids; do
    echo "Killing process with PID: $pid"
    kill -9 $pid
done

echo "All GPU processes have been killed."
