import torch.nn as nn
from torchvision import models


def get_mobilenet_norm_stats():
    weights_preset = models.MobileNet_V2_Weights.DEFAULT.transforms()
    return list(weights_preset.mean), list(weights_preset.std)

def build_model(num_classes=7, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model




def unfreeze_backbone(model, unfreeze_from_layer=14):
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True
    print(f"Unfroze backbone from layer {unfreeze_from_layer}")
    return model


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")


if __name__ == "__main__":
    model = build_model(freeze_backbone=True)
    count_params(model)

    model = unfreeze_backbone(model, unfreeze_from_layer=14)
    count_params(model)