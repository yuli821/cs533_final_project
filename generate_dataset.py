import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 원하는 경로에 저장 (예: /home/jykang5/datasets/)
root = "/projects/beih/yuli9/datasets"

datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
