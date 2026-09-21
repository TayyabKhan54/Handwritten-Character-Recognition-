"""Quick smoke test: runs a single training step on a small MNIST subset."""
from data import get_dataloaders, small_subset
from model import create_model
import torch
import torch.nn as nn
import torch.optim as optim


def run():
    train_loader, test_loader, num_classes = get_dataloaders("MNIST", batch_size=64)
    small_train = small_subset(train_loader)
    device = torch.device("cpu")
    model = create_model(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for images, targets in small_train:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print("Smoke test step loss:", loss.item())
        break


if __name__ == "__main__":
    run()
