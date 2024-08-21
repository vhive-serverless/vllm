import torch
import torch.distributed as dist
import multiprocessing as mp
from utils import send_dict, receive_dict
import time
from profiler import CudaMemoryProfiler
from liquid_communicator import LiquidCommunicator

TCP_PORT = 12344
TCP_STORE_PORT = 12345
dtype = torch.float16
buffer_size_gb = 1

tensor_size = 1
tensor_num = 10
def main_send():
    try:
        # Initialize the process group for distributed communication
        torch.cuda.set_device(0)
        group = dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{TCP_PORT}', world_size=2, rank=0)

        # Allocate 10 tensors, each of 1GB on GPU
        tensor_length = int(tensor_size * (1024**3) / torch.finfo(dtype).bits * 8)  # Calculate number of elements
        tensor_dict = {f'tensor_{i}': torch.rand(tensor_length, dtype=dtype, device='cuda') for i in range(tensor_num)}

        comm = LiquidCommunicator(
            buffer_size_gb=buffer_size_gb,
            group=group,
            tcp_store_port=TCP_STORE_PORT,
            dtype=dtype,
        )

        # Send the tensor dictionary to another GPU
        dist.barrier()
        with CudaMemoryProfiler(interval=0.1,cuda_index=0) as m:
            comm.send_dict(tensor_dict, dst_rank=1)
        # dist.barrier()
        mem_records = m.get_memory_records()
        print(f"Peak memory during sending: {max(mem_records):.1f}GB")


        # received_dict = comm.recv_dict(src_rank=1, keys=list(tensor_dict.keys()))
        # assert received_dict.keys() == tensor_dict.keys()
        # for key, tensor in tensor_dict.items():
        #     assert received_dict[key].allclose(tensor)

    except KeyboardInterrupt:
        print(f"Send process terminated")

def main_receive():
    # Initialize the process group for distributed communication
    torch.cuda.set_device(1)
    group=dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{TCP_PORT}', world_size=2, rank=1)

    # Define the keys of the tensors you expect to receive
    keys = [f'tensor_{i}' for i in range(tensor_num)]

    # Create a TCPStore
    comm = LiquidCommunicator(
        buffer_size_gb=buffer_size_gb,
        group=group,
        tcp_store_port=TCP_STORE_PORT,
        dtype=dtype
    )

    # Receive the tensor dictionary from another GPU
    dist.barrier()
    start = time.time()
    with CudaMemoryProfiler(interval=0.01, cuda_index=1) as m:
        received_dict = comm.recv_dict(src_rank=0, keys=keys)
    # dist.barrier()
    latency = time.time() - start
    bw = tensor_num * tensor_size / latency
    print(f"Receive dict finished, latency: {latency:.2f}s, bandwidth: {bw:.1f}GB/s")
    mem_records = m.get_memory_records()
    print(f"Peak memory during recving: {max(mem_records):.1f}GB")

    # comm.send_dict(received_dict, 0) 

if __name__ == "__main__":
    # Create two processes for send and receive
    send_process = mp.Process(target=main_send)
    receive_process = mp.Process(target=main_receive)

    # Start the processes
    send_process.start()
    # time.sleep(1)
    receive_process.start()

    # Wait for both processes to complete
    send_process.join()
    receive_process.join()