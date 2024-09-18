#!/bin/bash
run_experiment() {
    pattern=$1  # 第一个参数作为 pattern 传入

    # 拼接 newname
    newname="conv_win5_${pattern}"

    # 执行更改内容的脚本，传入 pattern
    /home/lrq/proj/vllm/change_content.sh $pattern

    #start exp
    /home/lrq/anaconda3/envs/vllml/bin/python /home/lrq/proj/vllm/liquid_server.py


    /home/lrq/proj/vllm/kill_router.sh
    /home/lrq/proj/vllm/kill_gpu.sh

    /home/lrq/proj/vllm/change_name.sh $newname
}

run_experiment 10
run_experiment 20
run_experiment 30
run_experiment 40
run_experiment 50
# run_experiment 60
# run_experiment 70
# run_experiment 80
# run_experiment 90
# run_experiment 100
# run_experiment 110
# run_experiment 120
# run_experiment 130
