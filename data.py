import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch
from typing import Tuple


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_dataloaders(dataset: str = "MNIST", batch_size: int = 64, emnist_split: str = "letters") -> Tuple[DataLoader, DataLoader, int]:
    transform = get_transforms()
    if dataset.upper() == "MNIST":
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset.upper() == "EMNIST":
        # EMNIST has multiple splits; 'letters' is common for letters (labels 1..26)
        train_ds = datasets.EMNIST(root="./data", split=emnist_split, train=True, download=True, transform=transform)
        test_ds = datasets.EMNIST(root="./data", split=emnist_split, train=False, download=True, transform=transform)
        # letters split's labels are 1-26; we remap in training/eval if needed
        num_classes = 26 if emnist_split == "letters" else len(train_ds.classes)
    else:
        raise ValueError("Unsupported dataset: use MNIST or EMNIST")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, num_classes


def small_subset(loader: DataLoader, max_batches: int = 2) -> DataLoader:
    # Create a small subset loader for quick smoke tests
    ds = loader.dataset
    total = len(ds)
    take = min(256, total)
    subset = Subset(ds, list(range(take)))
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False)
