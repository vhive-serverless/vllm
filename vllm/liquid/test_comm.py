import torch
import torch.distributed as dist
import multiprocessing as mp
from utils import send_dict, receive_dict
import time

TCP_PORT = 12344
TCP_STORE_PORT = 12345

def main_send():
    # Initialize the process group for distributed communication
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{TCP_PORT}', world_size=2, rank=0)

    # Allocate 10 tensors, each of 1GB on GPU
    tensor_dict = {f'tensor_{i}': torch.empty(int(1 * 1024**3 / 4), dtype=torch.float16, device='cuda') for i in range(10)}

    # Create a TCPStore
    store = dist.TCPStore("localhost", TCP_STORE_PORT, world_size=2, is_master=True)

    # Send the tensor dictionary to another GPU
    dist.barrier()
    start = time.time()
    send_dict(tensor_dict, dst_rank=1, store=store)
    dist.barrier()
    latency = time.time() - start
    bw = 10 / latency
    print(f"Send dict finished, latency: {latency:.2f}s, bandwidth: {bw:.1f}GB/s")

def main_receive():
    # Initialize the process group for distributed communication
    torch.cuda.set_device(1)
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{TCP_PORT}', world_size=2, rank=1)

    # Define the keys of the tensors you expect to receive
    keys = [f'tensor_{i}' for i in range(10)]

    # Create a TCPStore
    store = dist.TCPStore("localhost", TCP_STORE_PORT, world_size=2, is_master=False)

    # Receive the tensor dictionary from another GPU
    dist.barrier()
    received_dict = receive_dict(src_rank=0, store=store, keys=keys)
    dist.barrier()

if __name__ == "__main__":
    # Create two processes for send and receive
    send_process = mp.Process(target=main_send)
    receive_process = mp.Process(target=main_receive)

    # Start the processes
    send_process.start()
    receive_process.start()

    # Wait for both processes to complete
    send_process.join()
    receive_process.join()