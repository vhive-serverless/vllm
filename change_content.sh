#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 new_format"
  exit 1
fi

filename="/home/lrq/proj/vllm/liquid_server.py"

newline="                '-pattern', 'azure-conv-"
medium=$1
suffix="-5',"
content="${newline}${medium}${suffix}"
# 替换第 59 行的内容
sed -i "59s/.*/${content}/" "$filename"

echo "Line 59 in $filename has been updated."
