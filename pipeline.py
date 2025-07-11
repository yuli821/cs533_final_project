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
import gc
import sys
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import argparse
from torch.cuda import cudart, check_error
import torch.cuda.nvtx as nvtx

# torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.automatic_dynamic_shapes = False
torch.set_float32_matmul_precision('high')
dataset = 'cifar100'
rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
image_size = 32
global_batch_size = 64
batch_size = global_batch_size // world_size
chunks = world_size

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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

        train_sampler = None
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    elif dataset == "cifar100":
        #training and test dataset
        trainset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/", train=True, transform=train_transform, download=True)
        testset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/", train=False, transform=test_transform, download=True)

        #train and test dataloaders
        train_sampler = None
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True)
        testloader = DataLoader(testset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
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

def main():
    # world_size = torch.cuda.device_count()
    use_optimization = True  # Toggle this flag to enable/disable comm optimization
    # model_size = 'huge16'      # Change to 'large' or 'huge' as needed
    num_epoch = 5
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description="Distributed ViT Training")
    parser.add_argument("model_size", choices=[
        'base4','large4','huge4',
        'base16','large16','huge16',
        'base32','large32','huge32'
    ], help="ViT model size to train")
    args, unknown = parser.parse_known_args()
    model_size = args.model_size # 'huge4'      # Change to 'large' or 'huge' as needed
    # print("\n[INFO] Running WITHOUT communication optimization\n")
    # mp.spawn(train, args=(world_size, model_size, num_epoch, False), nprocs=world_size, join=True) #data parallelism

    # print("\n[INFO] Running WITH communication optimization\n")
    # mp.spawn(train, args=(world_size, model_size, num_epoch, True), nprocs=world_size, join=True)
    
    #pipeline
    # rank = int(os.environ["SLURM_PROCID"])
    # world_size = int(os.environ["SLURM_NTASKS"])
    print(f"Rank: {rank}")
    print(f"World size: {world_size}")
    setup(rank, world_size)
    # Figure out device to use
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    print(device)

    nclasses = 100 if dataset == 'cifar100' else 10

    config = get_vit_config(model_size)
    assert config["mlp_dim"] % config["d_model"] == 0
    mlp_ratio = config["mlp_dim"] / config["d_model"]
    model = vit(ipch=3, image_size=image_size, Nclasses=nclasses, num_layers=config["num_layers"], patch_size=config["patch_size"],d_model=config["d_model"], nhead=config["nhead"], mlp_ratio=mlp_ratio).to(device)
    # trainloader, testloader, train_sampler = getdata(image_size=image_size, batch_size=global_batch_size, rank=rank, world_size=world_size)
    # if rank == 0:
    trainloader, testloader, train_sampler = getdata(
        dataset= dataset,
        image_size=image_size,
        batch_size=global_batch_size,
        rank=0,
        world_size=1  # Don't partition dataset
    )
    # else:
        # trainloader, testloader, train_sampler = None, None, None
    
    example_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if config["num_layers"] == 32:
        num_stages = 4
        split_spec = {
            'blocks.7': SplitPoint.END,
            'blocks.15': SplitPoint.END,
            'blocks.23': SplitPoint.END
        }
    elif config["num_layers"] == 24:
        num_stages = 4
        split_spec = {
            'blocks.5': SplitPoint.END,
            'blocks.11': SplitPoint.END,
            'blocks.17': SplitPoint.END
        }
    else:
        num_states = 3
        split_spec = {
            'blocks.3': SplitPoint.END,
            'blocks.7': SplitPoint.END
        }
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)
    del model
    if rank == 0:
        print(" pipe ".center(80, "*"))
        print(pipe)
        print(" stage 0 ".center(80, "*"))
        print(pipe.split_gm.submod_0)
        print(" stage 1 ".center(80, "*"))
        print(pipe.split_gm.submod_1)
        print(" stage 2 ".center(80, "*"))
        print(pipe.split_gm.submod_2)
        print(" stage 3 ".center(80, "*"))
        print(pipe.split_gm.submod_3)

    # stage = PipelineStage(pipe, rank, world_size, device)
    stage = pipe.build_stage(rank, device=device)
    print(f"[Rank {rank}] Submodule:")
    print(stage.submod)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(stage.submod.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1, t_total=num_epoch)
    schedule = ScheduleGPipe(stage, chunks, loss_fn=criterion)

    # Run the pipeline with input `x`. Divide the batch into 4 micro-batches
    # and run them in parallel on the pipeline
    dist.barrier()
    check_error(cudart().cudaProfilerStart())
    for epoch in range(num_epoch):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(rank)
        # train_sampler.set_epoch(epoch)
        # with profile(
        #     activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=True,
        #     with_modules=True,
        #     profile_memory=True,
        #     schedule=torch.profiler.schedule(wait=10, warmup=0, active=5, repeat=1),
        #     with_flops=True,
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'/work/hdd/beih/yuli9/log_pipe/{model_size}/rank_{rank}')
        # ) as prof:
        for b, (x, y) in enumerate(trainloader) :
            # x, y = x.to(device), y.to(device)
            if rank == 0: 
                x = x.to(device)
                # with record_function("rank0_pipeline_step"):
                nvtx.range_push(f"Rank {rank} - Forward Step")
                schedule.step(x)
                nvtx.range_pop()
                nvtx.range_push(f"Rank {rank} - Optimizer Step")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop()
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                print(f"GPU {device} Batch {b} -------- power usage= {power:.2f} Watts")
            elif rank == world_size - 1:
                y = y.to(device)
                # with record_function("rank3_pipeline_step"):
                nvtx.range_push(f"Rank {rank} - Forward Step")
                losses = []
                output = schedule.step(target=y, losses=losses)
                nvtx.range_pop()

                nvtx.range_push(f"Rank {rank} - Optimizer Step")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop()
                print(f"Epoch {epoch} Batch {b}")
                for loss in losses:
                    print(f"Loss: {loss.item():.4f}")
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                print(f"GPU {device} Batch {b} -------- power usage= {power:.2f} Watts")
            else:
                # with record_function(f"rank{rank}_pipeline_step"):
                nvtx.range_push(f"Rank {rank} - Forward Step")
                schedule.step()
                nvtx.range_pop()

                nvtx.range_push(f"Rank {rank} - Optimizer Step")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop()
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                print(f"GPU {device} Batch {b} -------- power usage= {power:.2f} Watts")
                # prof.step()
        scheduler.step()
        # for b, (x,y) in enumerate(trainloader) :
        #     x, y = x.to(device), y.to(device)
        #     # with record_function(f"pipeline_schedule_{rank}"):
        #     if rank == 0:
        #         nvtx.range_push(f"Rank {rank} - Forward Step")
        #         schedule.step(x)
        #         nvtx.range_pop()
        #         nvtx.range_push(f"Rank {rank} - Optimizer Step")
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none=True)
        #         nvtx.range_pop()
        #     elif rank == world_size - 1:
        #         nvtx.range_push(f"Rank {rank} - Forward Step")
        #         losses = []
        #         output = schedule.step(target=y, losses=losses)
        #         nvtx.range_pop()

        #         nvtx.range_push(f"Rank {rank} - Optimizer Step")
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none=True)
        #         nvtx.range_pop()
        #         print(f"Epoch {epoch} Batch {b}")
        #         for loss in losses:
        #             print(f"Loss: {loss.item():.4f}")
        #     else:
        #         nvtx.range_push(f"Rank {rank} - Forward Step")
        #         schedule.step()
        #         nvtx.range_pop()

        #         nvtx.range_push(f"Rank {rank} - Optimizer Step")
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none=True)
        #         nvtx.range_pop()
        #     prof.step()
        #     power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
    dist.barrier()
    check_error(torch.cuda.cudart().cudaProfilerStop())
    # torch.distributed.barrier()
    nvmlShutdown()
    if trainloader != None:
        del trainloader, testloader, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()