import math
from torch.optim.lr_scheduler import LambdaLR
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

    #This file uses pytorch implementation of dataloader and train the ViT model in single GPU
# Pytorch dataloader
def getdata(dataset='cifar10', image_size=32, batch_size=128, eval_batch_size=128, num_workers=4) :
    '''
    get desired training and inference dataloaders
    Inputs:
    dataset - name of dataset
    batch_size -
    num_workers - no. of parallel processes
    Outputs:
    trainloader -
    testloader -
    '''
    #pre-processing transforms
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

    if dataset == 'cifar10' :
        #training and test dataset
        trainset = datasets.CIFAR10(root="/projects/beih/yuli9/datasets/",
                                    train=True,
                                    transform=train_transform,
                                    download=True)
        testset = datasets.CIFAR10(root="/projects/beih/yuli9/datasets/",
                                  train=False,
                                  transform=test_transform,
                                  download=True)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        #train and test dataloaders
        trainloader = DataLoader(dataset=trainset,
                                batch_size=batch_size,
                                sampler = train_sampler,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers)
        testloader = DataLoader(dataset=testset,
                                batch_size=eval_batch_size,
                                sampler = test_sampler,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)
    elif dataset == "cifar100":
        #training and test dataset
        trainset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/",
                                    train=True,
                                    transform=train_transform,
                                    download=True)
        testset = datasets.CIFAR100(root="/projects/beih/yuli9/datasets/",
                                  train=False,
                                  transform=test_transform,
                                  download=True)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        #train and test dataloaders
        trainloader = DataLoader(dataset=trainset,
                                batch_size=batch_size,
                                sampler = train_sampler,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers)
        testloader = DataLoader(dataset=testset,
                                batch_size=eval_batch_size,
                                sampler = test_sampler,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)
    else :
        print("Dataset not defined")
        trainloader, testloader = None, None
    return trainloader, testloader