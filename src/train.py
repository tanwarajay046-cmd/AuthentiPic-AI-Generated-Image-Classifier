"""Train the ConvNeXt-Base classifier on the AI vs. Human-Generated Images dataset.

Expected data layout (see README for download instructions):

    data/
      ai-vs-human-generated-dataset/   # image files referenced by train.csv / test.csv
      train.csv
      test.csv

Usage:
    python src/train.py
"""

import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AIImageDataset, InferenceImageDataset, train_transforms, val_test_transforms
from model import build_model, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AI-vs-human image classifier.")
    parser.add_argument("--images-dir", default="data/ai-vs-human-generated-dataset")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", default="model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pd.read_csv(args.train_csv)
    train = train[["file_name", "label"]]
    train.columns = ["id", "label"]

    train_df, val_df = train_test_split(
        train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train["label"],
    )

    train_dataset = AIImageDataset(train_df, root_dir=args.images_dir, transform=train_transforms)

    val_file_list = [f"{args.images_dir}/{fname}" for fname in val_df["id"]]
    val_labels = val_df["label"].values
    val_dataset = InferenceImageDataset(file_list=val_file_list, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(device)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = output.argmax(dim=1)
            epoch_accuracy += (preds == label).float().mean().item()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_pred_classes, val_labels_list = [], []

        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")):
                data = data.to(device)
                output = model(data)

                batch_labels = val_labels[i * val_loader.batch_size:(i + 1) * val_loader.batch_size]
                batch_labels = torch.tensor(batch_labels, device=device)

                loss = criterion(output, batch_labels)
                val_loss += loss.item()

                preds = output.argmax(dim=1)
                val_acc += (preds == batch_labels).float().mean().item()

                val_pred_classes.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 = f1_score(val_labels_list, val_pred_classes, average="binary")

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        scheduler.step()

    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"Saved model weights to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
