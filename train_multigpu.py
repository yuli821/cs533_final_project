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
from utils import WarmupLinearSchedule
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
# torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.automatic_dynamic_shapes = False
torch.set_float32_matmul_precision('high')
dataset = 'cifar100'
image_size = 32
batch_size = 64
rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def getdata(dataset='cifar10', image_size=32, batch_size=batch_size, num_workers=4, rank=0, world_size=1):
    train_transform = transforms.Compose([transforms.RandomCrop(image_size, image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2471, 0.2435, 0.2616]),
                                            ])
    test_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2471, 0.2435, 0.2616]),
                                            ])
    if dataset == 'cifar10':
        trainset = datasets.CIFAR10(root="/projects/beih/yuli9/datasets", train=True, transform=train_transform, download=False)
        testset = datasets.CIFAR10(root="/projects/beih/yuli9/datasets", train=False, transform=test_transform, download=False)

        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    elif dataset == "cifar100":
        #training and test dataset
        trainset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/", train=True, transform=train_transform, download=True)
        testset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/", train=False, transform=test_transform, download=True)

        #train and test dataloaders
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler, shuffle=True, pin_memory=True, num_workers=num_workers)
        testloader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    else:
        raise ValueError("Dataset not supported")

    return trainloader, testloader, train_sampler

def get_vit_config(model_size):
    if model_size == 'base4':
        return {"num_layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 4}
    elif model_size == 'large4':
        return {"num_layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 4}
    elif model_size == 'huge4':
        return {"num_layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 4}
    elif model_size == 'base16':
        return {"num_layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 16}
    elif model_size == 'large16':
        return {"num_layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 16}
    elif model_size == 'huge16':
        return {"num_layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 16}
    elif model_size == 'base32':
        return {"num_layers": 12, "d_model": 768, "mlp_dim": 3072, "nhead": 12, "patch_size": 32}
    elif model_size == 'large32':
        return {"num_layers": 24, "d_model": 1024, "mlp_dim": 4096, "nhead": 16, "patch_size": 32}
    elif model_size == 'huge32':
        return {"num_layers": 32, "d_model": 1280, "mlp_dim": 5120, "nhead": 16, "patch_size": 32}
    else:
        raise ValueError("Unsupported ViT model size")

def trainepoch(model, epoch, trainloader, criterion, opt, device, rank, run_name) :
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(rank)

    epochloss = 0
    acc = 0
    total = 0
    if epoch == 1:
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'/u/yuli9/cs533_final_project/log/{run_name}/rank_{rank}')
        ) as prof:
            for b, (x,y) in enumerate(trainloader) :
                x, y = x.to(device), y.to(device)

                #forward pass
                with record_function("forward"):
                    logits = model(x)
                predicted = torch.argmax(logits, dim=-1)
                acc += torch.eq(predicted, y).sum()
                total += y.shape[0]
                #backward pass
                loss = criterion(logits, y)
                with record_function("backward"):
                    loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                prof.step()
                #accumulated loss
                print("GPU ", device, " ", "Batch ", b, "--------- loss = ", loss.item())

                # TODO: print power consumption perf
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                print(f"GPU {device} Batch {b} -------- power usage= {power:.2f} Watts")

                epochloss += loss.item()
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # TODO: repeat what if statement profiler is doing
    else: 
        for b, (x,y) in enumerate(trainloader) :
            x, y = x.to(device), y.to(device)

            #forward pass
            logits = model(x)
            predicted = torch.argmax(logits, dim=-1)
            acc += torch.eq(predicted, y).sum()
            total += y.shape[0]
            #backward pass
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            #accumulated loss
            print("GPU ", device, " ", "Batch ", b, "--------- loss = ", loss.item())

            # TODO: print power consumption perf
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
            print(f"GPU {device} Batch {b} -------- power usage= {power:.2f} Watts")

            epochloss += loss.item()
    nvmlShutdown()
    return model, epochloss/b, acc/total

def train(rank, world_size, model_size='base16', num_epoch=5, use_optimization=False):
    run_name = f"{model_size}_{'opt' if use_optimization else 'noopt'}"
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print("Beginning training on ", device)

    nclasses = 100 if dataset == 'cifar100' else 10

    trainloader, testloader, train_sampler = getdata(image_size=image_size, batch_size=batch_size, rank=rank, world_size=world_size)

    config = get_vit_config(model_size)
    assert config["mlp_dim"] % config["d_model"] == 0
    mlp_ratio = config["mlp_dim"] / config["d_model"]
    model = vit(ipch=3, image_size=image_size, Nclasses=nclasses, num_layers=config["num_layers"], patch_size=config["patch_size"],d_model=config["d_model"], nhead=config["nhead"], mlp_ratio=mlp_ratio).to(device)

    if use_optimization:
        # model = DDP(model, device_ids=[rank], bucket_cap_mb=25, find_unused_parameters=True)
        # torch._dynamo.config.optimize_ddp: general optimization on operator fusion, loop unrolling, and memory reuse inside DDP
        # gradient_as_bucket_view: optimization on communication
        torch._dynamo.config.optimize_ddp = True
        model = torch.compile(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        # torch._dynamo.config.optimize_ddp: general optimization on operator fusion, loop unrolling, and memory reuse inside DDP
        # gradient_as_bucket_view: optimization on communication
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, weight_decay=0.3)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1, t_total=num_epoch)

    model.train()
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        model, loss, acc = trainepoch(model, epoch, trainloader, criterion, optimizer, device, rank, run_name)
        scheduler.step()
        if rank == 0:
            print(f"[GPU {rank}] Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    if rank == 0:
        torch.save(model.module.state_dict(), f"/u/yuli9/cs533_final_project/models/{dataset}/final_{model_size}.pt")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    use_optimization = True  # Toggle this flag to enable/disable comm optimization
    model_size = 'huge16'      # Change to 'large' or 'huge' as needed
    num_epoch = 5
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    setup(rank, world_size)
    print("\n[INFO] Running WITHOUT communication optimization\n")
    mp.spawn(train, args=(world_size, model_size, num_epoch, False), nprocs=world_size, join=True)
    # train(rank, world_size, model_size, num_epoch, False)

    # print("\n[INFO] Running WITH communication optimization\n")
    # mp.spawn(train, args=(world_size, model_size, num_epoch, True), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()