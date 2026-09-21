import argparse
import torch
import torch.nn as nn
from data import get_dataloaders
from model import create_model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def evaluate_checkpoint(checkpoint, dataset: str, emnist_split: str, batch_size: int, device):
    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size, emnist_split)
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
    return loss_sum / len(test_loader.dataset), correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "EMNIST"])
    parser.add_argument("--emnist-split", default="letters")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    val_loss, val_acc = evaluate_checkpoint(ckpt, args.dataset, args.emnist_split, args.batch_size, device)
    print(f"Eval: loss={val_loss:.4f} acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
