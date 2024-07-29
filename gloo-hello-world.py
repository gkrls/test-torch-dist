#!/usr/bin/env python
import os
import torch
import torch.distributed as dist

def run(rank, size):
    """ Distributed function to be implemented later. """
    print("[RANK %d] Hello from run()" % rank)

os.environ['MASTER_ADDR'] = '42.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['GLOO_SOCKET_IFNAME'] = 'ens4f0'

if __name__ == "__main__":
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD'])

    dist.init_process_group('gloo', rank=rank, world_size=world)
    run(rank, world)