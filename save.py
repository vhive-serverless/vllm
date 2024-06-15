import argparse
import torch
from transformers import AutoModelForCausalLM
from checkpoint_store import save_model

# Set up argument parser
parser = argparse.ArgumentParser(description='Load and save a HuggingFace model.')
parser.add_argument('--model-name', type=str, required=True, help='Name of the model to load from HuggingFace model hub.')

# Parse arguments
args = parser.parse_args()
model_name = args.model_name

# Load the model from HuggingFace model hub
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Replace './models' with your local path
save_model(model, f'./models/{model_name}')
