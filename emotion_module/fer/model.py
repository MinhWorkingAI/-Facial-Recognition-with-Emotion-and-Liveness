import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7

def build_model(freeze_backbone=True):
    # Load MobileNetV2 with ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head with our own
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


def unfreeze_backbone(model, unfreeze_from_layer=14):
    """
    Partially unfreeze the backbone for fine-tuning.
    MobileNetV2 has 18 feature layers (0–17).
    Unfreezing from layer 14 onwards is a good balance.
    """
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    print(f"Unfroze backbone from layer {unfreeze_from_layer} onwards.")
    return model


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")


if __name__ == "__main__":
    model = build_model(freeze_backbone=True)
    count_params(model)
    print(model.classifier)