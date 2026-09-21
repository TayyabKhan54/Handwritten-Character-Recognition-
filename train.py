import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataloaders
from model import create_model


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


def evaluate(model, device, loader, criterion):
    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "EMNIST"])
    parser.add_argument("--emnist-split", default="letters")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    args = parser.parse_args()

    train_loader, test_loader, num_classes = get_dataloaders(args.dataset, args.batch_size, args.emnist_split)

    device = torch.device(args.device)
    model = create_model(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
    }, args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
