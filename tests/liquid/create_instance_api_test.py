import uuid
import requests

# Define the URL
url = "http://localhost:8001/create_instance"

# Generate a UUID
generated_uuid = str(uuid.uuid4())

# Construct the JSON payload
payload = {
    "uuid": generated_uuid,
    "cli": "vllm",
    "env": ["CUDA_VISIBLE_DEVICES=0,1"],
    "args": [
        "serve",
        "facebook/opt-125m",
        "--gpu-memory-utilization",
        "0.9",
        "--load-format",
        "auto",
        "--served-model-name",
        "facebook/opt-125m",
        "--enable-chunked-prefill",
        "True",
        "--max-num-batched-token",
        "1024",
        "--tensor-parallel-size",
        "2",
        "--host",
        "0.0.0.0",
        "--port",
        "23333"
    ]
}

# Send the POST request
response = requests.post(url, json=payload)

# Output response for verification
print(f"Status Code: {response.status_code}")
print(f"Response Body:\n{response.text}")