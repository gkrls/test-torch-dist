#!/usr/bin/env python
import os
import torch
import torch.distributed as dist


def get_available_devices(rank):
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"[rank {rank}] Number of available CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"[rank {rank}] Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"[rank {rank}] CUDA is not available. Only CPU is available.")

def run(rank, size):
    """ Distributed function to be implemented later. """
    get_available_devices(rank)

os.environ['MASTER_ADDR'] = '42.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['GLOO_SOCKET_IFNAME'] = 'ens4f0'

if __name__ == "__main__":
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD'])

    dist.init_process_group('gloo', rank=rank, world_size=world)
    run(rank, world)