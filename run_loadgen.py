import subprocess
model_name = "facebook/opt-6.7b"
command = [
        './LLMLoadgen',
        '-pattern', 'azure-conv-20-5',
        '-dataset', 'azure-conv',
        '-dst', 'liquid',
        '-ip', 'localhost',
        '-port', '8000',
        # '-limit', '100',
        '-max_drift', '100',
        '-model_name', f'{model_name}'
    ]

working_dir = '/home/lrq619/proj/vllm/LLMLoadgen/release'
result = subprocess.run(command, cwd=working_dir, capture_output=True)