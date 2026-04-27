"""Torchvision image transforms for aligned plant-disease classification (train vs eval)."""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics (pretrained ResNet / EfficientNet backbones expect this scaling).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmented transforms for training PlantVillage (light geometry + color jitter + ImageNet norm)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.12, contrast=0.12, saturation=0.12, hue=0.04
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation / PlantDoc evaluation (resize + ImageNet norm)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
