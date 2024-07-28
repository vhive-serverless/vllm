import argparse
import os
from typing import Optional
# Set up argument parser
parser = argparse.ArgumentParser(description='Load and save a HuggingFace model.')
parser.add_argument('--model-name', type=str, required=True, help='Name of the model to load from HuggingFace model hub.')

# Parse arguments
args = parser.parse_args()
model_name = args.model_name


class VllmModelDownloader:

    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        import shutil
        import torch
        import gc
        from tempfile import TemporaryDirectory
        from huggingface_hub import snapshot_download
        from vllm import LLM

        def _run_writer(input_dir, output_dir):
            llm_writer = LLM(
                model=input_dir,
                download_dir=input_dir,
                dtype=torch_dtype,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="mp",
            )
            model_executer = llm_writer.llm_engine.model_executor
            # TODO: change the `save_sharded_state` to `save_serverless_llm_state`
            model_executer.save_serverless_llm_state(
                path=output_dir, pattern=pattern, max_size=max_size
            )
            for file in os.listdir(input_dir):
                if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(output_dir, file)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy(src_path, output_dir)
            del model_executer
            del llm_writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        storage_path = os.getenv("MODEL_PATH", "./models")
        output_dir = os.path.join(storage_path, model_name)
        # create the output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = snapshot_download(model_name, cache_dir=cache_dir)
                _run_writer(input_dir, output_dir)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            shutil.rmtree(output_dir)
            raise e


torch_dtype = "float16"
downloader = VllmModelDownloader()
downloader.download_vllm_model(model_name, torch_dtype, 1)