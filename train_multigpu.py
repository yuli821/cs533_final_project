import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ViT import vit
from torch.profiler import profile, record_function, ProfilerActivity
torch._dynamo.config.optimize_ddp = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def getdata(dataset='cifar10', batch_size=128, num_workers=4, rank=0, world_size=1):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
        ])

        trainset = datasets.CIFAR10(root="../datasets/", train=True, transform=train_transform, download=True)
        testset = datasets.CIFAR10(root="../datasets/", train=False, transform=test_transform, download=True)

        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError("Dataset not supported")

    return trainloader, testloader, train_sampler

def get_vit_config(model_size):
    if model_size == 'base4':
        return {"layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 4}
    elif model_size == 'large4':
        return {"layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 4}
    elif model_size == 'huge4':
        return {"layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 4}
    elif model_size == 'base16':
        return {"layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 16}
    elif model_size == 'large16':
        return {"layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 16}
    elif model_size == 'huge16':
        return {"layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 16}
    elif model_size == 'base32':
        return {"layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 32}
    elif model_size == 'large32':
        return {"layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 32}
    elif model_size == 'huge32':
        return {"layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 32}
    else:
        raise ValueError("Unsupported ViT model size")

def trainepoch(model, num_layers, trainloader, criterion, opt, device, rank) :
    epochloss = 0
    acc = 0
    total = 0
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/rank_{rank}')
    ) as prof:
        for b, (x,y) in enumerate(trainloader) :
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            #forward pass
            with record_function("forward"):
                logits = model(x, num_layers)
            predicted = torch.argmax(logits, dim=-1)
            acc += torch.eq(predicted, y).sum()
            total += y.shape[0]
            #backward pass
            loss = criterion(logits, y)
            with record_function("backward"):
                loss.backward()
            opt.step()
            prof.step()
            #accumulated loss
            print("GPU ", device, " ", "Batch ", b, "--------- loss = ", loss.item())
            epochloss += loss
    return model, epochloss/b, acc/total

# def trainepoch(model, num_layers, trainloader, criterion, opt, device):
#     model.train()
#     epochloss, acc, total = 0, 0, 0
#     for x, y in trainloader:
#         x, y = x.to(device), y.to(device)
#         opt.zero_grad()
#         logits = model(x, num_layers)
#         predicted = torch.argmax(logits, dim=-1)
#         acc += (predicted == y).sum()
#         total += y.size(0)
#         loss = criterion(logits, y)
#         loss.backward()
#         opt.step()
#         epochloss += loss.item()
#     return epochloss / len(trainloader), acc.item() / total

def train(rank, world_size, model_size='base16', num_epoch=5, use_optimization=False):
    setup(rank, world_size)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(f"cuda:{rank}")

    trainloader, testloader, train_sampler = getdata(batch_size=128, rank=rank, world_size=world_size)

    config = get_vit_config(model_size)
    assert config["mlp_dim"] % config["d_model"] == 0
    mlp_ratio = config["mlp_dim"] / config["d_model"]
    model = vit(ipch=3, image_size=32, patch_size=config["patch_size"],d_model=config["d_model"], nhead=config["nhead"], mlp_ratio=mlp_ratio).to(device)

    if use_optimization:
        model = DDP(model, device_ids=[rank], bucket_cap_mb=25, find_unused_parameters=False)
    else:
        model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

    model.train()
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        model, loss, acc = trainepoch(model, config["num_layers"], trainloader, criterion, optimizer, device, rank)
        scheduler.step()
        if rank == 0:
            print(f"[GPU {rank}] Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    if rank == 0:
        torch.save(model.module.state_dict(), f"../models/cifar10/final_{model_size}.pt")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    use_optimization = True  # Toggle this flag to enable/disable comm optimization
    model_size = 'base'      # Change to 'large' or 'huge' as needed
    num_epoch = 5
    mp.spawn(train, args=(world_size, model_size, num_epoch, use_optimization), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()