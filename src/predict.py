"""Run inference with a trained checkpoint and produce a Kaggle-style submission.csv.

Usage:
    python src/predict.py --checkpoint model.pt
"""

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import InferenceImageDataset, val_test_transforms
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for the test set.")
    parser.add_argument("--images-dir", default="data/ai-vs-human-generated-dataset")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test = pd.read_csv(args.test_csv)
    test_file_list = [f"{args.images_dir}/{fname}" for fname in test["id"]]
    test_dataset = InferenceImageDataset(file_list=test_file_list, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_logits, test_pred_classes = [], []
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Generating predictions"):
            data = data.to(device)
            output = model(data)
            test_logits.extend(output.cpu().numpy())
            test_pred_classes.extend(output.argmax(dim=1).cpu().numpy())

    logits_df = pd.DataFrame(test_logits, columns=["logit_class_0", "logit_class_1"])
    logits_df["id"] = test["id"].values
    logits_df.to_csv("test_logits.csv", index=False)

    test["label"] = test_pred_classes
    test[["id", "label"]].to_csv(args.output, index=False)

    print("Saved logits to test_logits.csv")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
