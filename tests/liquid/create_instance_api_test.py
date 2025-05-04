import uuid
import requests
import time
create_url = "http://localhost:8001/create_instance"
delete_url = "http://localhost:8001/delete_instance"

def create_instance(model, gpu_ids, port):

    # Convert list of ints to comma-separated string for CUDA_VISIBLE_DEVICES
    cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"

    # Generate a UUID
    generated_uuid = str(uuid.uuid4())

    # Construct the JSON payload
    payload = {
        "uuid": generated_uuid,
        "cli": "vllm",
        "env": [cuda_env],
        "args": [
            "serve",
            model,
            "--gpu-memory-utilization",
            "0.9",
            "--load-format",
            "auto",
            "--served-model-name",
            model,
            "--enable-chunked-prefill",
            "True",
            "--max-num-batched-token",
            "1024",
            "--tensor-parallel-size",
            str(len(gpu_ids)),
            "--host",
            "0.0.0.0",
            "--port",
            str(port)
        ]
    }

    response = requests.post(create_url, json=payload)
    response.raise_for_status()  # Raise an error if the request failed

    return generated_uuid

def wait_for_http_service(port, timeout=60, interval=0.5):
    """Wait until an HTTP service becomes available at localhost:port.

    Args:
        port (int): The port to check.
        timeout (float): Maximum time to wait in seconds.
        interval (float): Time between checks in seconds.

    Raises:
        TimeoutError: If the service does not become available in time.
    """
    url = f"http://localhost:{port}/version"
    start_time = time.time()

    while True:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code < 500:
                return  # Service is up
        except requests.RequestException:
            pass  # Ignore connection errors

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service did not start on port {port} within {timeout} seconds.")

        time.sleep(interval)

def delete_instance(instance_uuid):

    payload = {"uuid": instance_uuid}
    response = requests.post(delete_url, json=payload)


def test_case():
    gpu_ids = [0,1]
    port = 23335
    instance_uuid = create_instance("facebook/opt-125m", gpu_ids, port)
    print(f"instance created with {instance_uuid}, waiting for http service to be ready on port: {port}")
    wait_for_http_service(port=port)
    print(f"Http service on port: {port} is ready! Delete the instance!")
    delete_instance(instance_uuid)

    # create another model
    gpu_ids = [1]
    port = 23336
    instance_uuid = create_instance("facebook/opt-350m", gpu_ids, port)
    print(f"instance created with {instance_uuid}, waiting for http service to be ready on port: {port}")
    wait_for_http_service(port=port)
    print(f"Http service on port: {port} is ready! Delete the instance!")
    delete_instance(instance_uuid)

if __name__ == "__main__":
    test_case()