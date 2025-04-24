To apply vllm patch:
```
conda activate vllm
pip install -e .
```
Then
```
export VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(os.path.abspath(vllm.__file__)))")
export PATCH_FILE=/home/lrq/proj/vllm/sllm_load.patch 
./patch.sh
```