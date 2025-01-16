import os
import sys
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler


def cleanup():
    # Destroy the process group
    dist.destroy_process_group()


def train(rank, params):
    print(f"Starting training on rank {rank}/pid {os.getpid()}...")

    # Ensure only rank 0 downloads the dataset
    if rank == 0:
        datasets.CIFAR10(root='./data', train=True, download=True)

    # Synchronize all processes to ensure the dataset is downloaded
    dist.barrier()

    # Define dataset and DataLoader with DistributedSampler
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    sampler = DistributedSampler(train_dataset, num_replicas=params['world_size'], rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, shuffle=False, num_workers=params['world_size'], pin_memory=True)

    device = params['device']
    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    ddp_model = DDP(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)

    # Training loop
    for epoch in range(params['epochs']):
        sampler.set_epoch(epoch)
        ddp_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if epoch == 0 and batch_idx == 0:
                print(f"Rank {rank} inputs: ", inputs[0].shape, 'R:', inputs[0][0]
                    [0][:3], "g:", inputs[0][1][0][:3], 'b:', inputs[0][2][0][:3])
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.cuda.synchronize()

            optimizer.step()
            running_loss += loss.item()
            # if rank == 0:
            if batch_idx % 200 == 0:
                print(
                    f"Rank {rank}, Epoch [{epoch+1}/{param['epochs']}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


def main(rank, params):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.environ['MASTER_ADDR'] = params['master_addr']
    os.environ['MASTER_PORT'] = params['master_port']
    dist.init_process_group(rank=rank, world_size=params['world_size'], backend=params['backend'])
    train(rank, params)
    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=2)
    # or your machine's IP address
    parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()


    if args.device == 'cuda' and torch.cuda.device_count() < 2 and params['backend'] == 'nccl':
        print("error: Cannot use NCCL backend with a single GPU. Please change the backend to 'gloo'")
        sys.exit(1)

    params = {
        "world_size": args.world,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "device": args.device,
        "backend": args.backend,
        "epochs": 2,
        "batch_size": 32
    }

    # Spawn processes
    mp.spawn(main, args=([params]), nprocs=args.world, join=True)
