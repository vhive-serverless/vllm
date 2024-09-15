#!/bin/bash

kill_processes_by_name() {
    process_name="$1"
    pids=$(ps aux | grep $process_name | grep -v grep | awk '{print $2}')

    if [ -z "$pids" ]; then
        echo "No processes named $process_name found."
    else
        for pid in $pids; do
            echo "Killing process with PID: $pid"
            kill -9 $pid
        done
        echo "All processes named $process_name have been killed."
    fi
}

kill_processes_by_name "LLMLoadgen"
