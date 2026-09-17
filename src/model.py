"""Model definition: a ConvNeXt-Base backbone with a custom binary classifier head."""

import torch
import torch.nn as nn
import torchvision.models as models


def build_model(device=None):
    """Build the ConvNeXt-Base classifier used for training and inference.

    Only the last two feature stages are fine-tuned; earlier stages stay frozen
    with their ImageNet-pretrained weights.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.convnext_base(weights="DEFAULT")

    # Freeze all backbone layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last two stages for fine-tuning
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Replace the classifier head
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2),
    )

    return model.to(device)


def build_optimizer(model, backbone_lr=1e-5, head_lr=1e-4):
    return torch.optim.AdamW([
        {"params": model.features[-2:].parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ])
