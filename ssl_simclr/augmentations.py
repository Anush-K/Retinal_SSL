"""
augmentations.py — SSL augmentation transforms (v3 — collapse fix)

Root cause of collapse in v2:
  base_aug and base_aug2 were identical pipelines. With a warm-started backbone,
  making two mildly-cropped versions of the same face look similar is trivially
  easy. Model solved it in ~5 epochs (L_within -> 0.006) and stopped learning.

Fix — asymmetric augmentation (standard in MoCo-v3, SimCLR-v2):
  view_s1 : "easy" anchor — mild crop, mild jitter.
  view_s2 : "hard" positive — aggressive crop, strong jitter, grayscale, solarize.
  view_f  : frequency view — RGB highpass per channel (unchanged from v2).

If matching two views is trivially easy, gradients vanish and model collapses.
Making view_s2 genuinely harder keeps the task non-trivial throughout training.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class HighPassRGB:
    """
    High-pass filter applied independently to each RGB channel.
    Per-channel (not grayscale) so color-domain frequency artifacts
    from deepfake blending/upsampling are preserved.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 0.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.float32)
        out = np.zeros_like(img_np)

        for c in range(3):
            channel = img_np[:, :, c]
            blurred = cv2.GaussianBlur(
                channel,
                (self.kernel_size, self.kernel_size),
                self.sigma
            )
            high = channel - blurred
            ch_min, ch_max = high.min(), high.max()
            if ch_max - ch_min > 1e-6:
                high = (high - ch_min) / (ch_max - ch_min) * 255.0
            else:
                high = np.zeros_like(high)
            out[:, :, c] = high

        return Image.fromarray(out.astype(np.uint8))


class RandomSolarize:
    """
    Randomly solarize with probability p.
    Inverts pixel values above threshold — forces model to use structure
    not raw pixel values. From SimCLR-v2.
    """
    def __init__(self, threshold: int = 128, p: float = 0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return TF.solarize(img, self.threshold)
        return img


def get_ssl_transforms(image_size: int = 224):
    """
    Returns (base_aug, hard_aug, highpass_aug) for DBFC training.

    base_aug    — easy anchor: mild crop (0.8-1.0), moderate jitter
    hard_aug    — hard positive: aggressive crop (0.4-1.0), strong jitter,
                  grayscale p=0.3, solarize p=0.2
    highpass_aug — frequency view: mild crop + RGB highpass

    WHY asymmetric: both views being "easy" -> trivial matching -> collapse.
    Easy anchor + hard positive -> model must learn real structure to match them.
    This is the MoCo-v3 / DINO augmentation strategy.
    """

    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    # View S1: easy anchor
    base_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        normalize,
    ])

    # View S2: hard positive — aggressive augmentation prevents trivial collapse
    hard_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        T.RandomGrayscale(p=0.3),
        RandomSolarize(threshold=128, p=0.2),
        T.ToTensor(),
        normalize,
    ])

    # View F: frequency-enhanced — mild spatial aug then RGB highpass
    highpass_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        HighPassRGB(kernel_size=5),
        T.ToTensor(),
        normalize,
    ])

    return base_aug, hard_aug, highpass_aug