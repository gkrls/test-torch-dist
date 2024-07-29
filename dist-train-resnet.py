#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # Single GPU per node
    else:
        return torch.device('cpu')


def train(rank, world):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True)

    device = get_device()
    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    ddp_model = DDP(model, device_ids=[device])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        ddp_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0 and rank == 0:
                print(
                    f"Rank {rank}, Epoch [{epoch+1}/{10}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int)
    parser.add_argument("--backend", choices=['gloo', 'nccl'], default='gloo')
    parser.add_argument("--master", default="42.0.0.1:29500")
    parser.add_argument("--iface", default="ens4f0")

    args = parser.parse_args()
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = args.master.split(':')[0]
        os.environ['MASTER_PORT'] = args.master.split(':')[1]
    if 'GLOO_SOCKET_IFNAME' not in os.environ:
        os.environ['GLOO_SOCKET_IFNAME'] = args.iface

    print(f"[rank {args.rank}] world: {args.world}, backend: {args.backend}, iface: {os.environ['GLOO_SOCKET_IFNAME']}, master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    # rank, world = int(os.environ['RANK']), int(os.environ['WORLD'])
    dist.init_process_group(
        args.backend, rank=args.rank, world_size=args.world)
    train(args.rank, args.world)
    dist.destroy_process_group()
