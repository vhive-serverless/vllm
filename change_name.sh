#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 new_format"
  exit 1
fi

cd "/home/lrq/proj/vllm"
new_filename="$1"
prefix="liquid_results_"
new_full_name="${prefix}${new_filename}.json"
touch liquid_results.json
mv liquid_results.json $new_full_name

cd "/home/lrq/proj/vllm"
prefix="output_"
new_full_name="${prefix}${new_filename}.txt"
touch output.txt
mv output.txt $new_full_name
