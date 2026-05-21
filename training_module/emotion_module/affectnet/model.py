import torch.nn as nn
from torchvision import models


def get_efficientnetv2s_norm_stats():
    weights_preset = models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
    return list(weights_preset.mean), list(weights_preset.std)


def get_mobilenet_norm_stats():
    # Backward-compatible alias for older scripts.
    return get_efficientnetv2s_norm_stats()

def build_model(num_classes=7, freeze_backbone=True):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    return model




def unfreeze_backbone(model, unfreeze_from_layer=5):
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

    model = unfreeze_backbone(model, unfreeze_from_layer=5)
    count_params(model)