"""
augmentations.py — LSFC augmentation transforms for retinal fundus SSL

Departure from DBFC (deepfake) augmentations:
1. Frequency branch: DBFC used a single Gaussian high-pass band (kernel=5)
   tuned to catch generative-upsampling artifacts. Retinal pathology has no
   such artifact — instead, DR lesions have well-documented multi-scale
   frequency structure (microaneurysms are fine-scale, ~2-5px; hemorrhages/
   exudates are coarse-scale, ~10-30px), consistent with the wavelet-based
   DR detection literature (Quellec et al. 2008; multi-resolution DWT-CSLBP
   approaches). We therefore decompose into two explicit bands instead of one.

2. Spatial augmentations: DBFC's hard positive included RandomGrayscale and
   solarization — reasonable for face images where color carries little
   forensic signal, but color IS a primary diagnostic signal in fundus
   photography (hemorrhage red vs. exudate yellow-white vs. background).
   We drop both and replace them with augmentations that reflect *actual*
   fundus camera acquisition variability instead: illumination vignetting
   and glare/reflection spots, both well-documented fundus imaging artifacts.

3. A "legacy" single-band transform (kernel=5, identical to the DBFC
   deepfake setting) is retained ONLY for the single-band ablation baseline,
   so that baseline is a faithful replication of the original DBFC design
   applied to retinal data — not a strawman.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import random


class HighPassRGB:
    """Per-channel Gaussian high-pass filter at a given kernel scale."""

    def __init__(self, kernel_size: int = 5, sigma: float = 0.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.float32)
        out = np.zeros_like(img_np)

        for c in range(3):
            channel = img_np[:, :, c]
            blurred = cv2.GaussianBlur(
                channel, (self.kernel_size, self.kernel_size), self.sigma
            )
            high = channel - blurred
            ch_min, ch_max = high.min(), high.max()
            if ch_max - ch_min > 1e-6:
                high = (high - ch_min) / (ch_max - ch_min) * 255.0
            else:
                high = np.zeros_like(high)
            out[:, :, c] = high

        return Image.fromarray(out.astype(np.uint8))


class IlluminationVignette:
    """
    Simulates radial illumination falloff ("vignetting"), a well-documented
    fundus camera acquisition artifact caused by uneven pupil illumination.
    Darkens image toward a randomly-offset edge, preserving color ratios
    (unlike solarize/grayscale, which would destroy diagnostic color cues).
    """

    def __init__(self, strength_range=(0.15, 0.4), p: float = 0.4):
        self.strength_range = strength_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img_np = np.array(img, dtype=np.float32)
        h, w = img_np.shape[:2]

        cx = w / 2 + random.uniform(-0.15, 0.15) * w
        cy = h / 2 + random.uniform(-0.15, 0.15) * h
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        strength = random.uniform(*self.strength_range)
        mask = 1.0 - strength * (dist / max_dist)
        mask = np.clip(mask, 1.0 - strength, 1.0)

        out = img_np * mask[:, :, None]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class GlareSpot:
    """
    Simulates a small bright reflection/glare spot — a common fundus
    photography artifact from flash reflection off the cornea or lens.
    """

    def __init__(self, p: float = 0.3, radius_frac=(0.03, 0.08), max_spots=2):
        self.p = p
        self.radius_frac = radius_frac
        self.max_spots = max_spots

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img_np = np.array(img, dtype=np.float32)
        h, w = img_np.shape[:2]
        n_spots = random.randint(1, self.max_spots)

        for _ in range(n_spots):
            cx = random.uniform(0.1, 0.9) * w
            cy = random.uniform(0.1, 0.9) * h
            r = random.uniform(*self.radius_frac) * min(h, w)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            glare_mask = np.clip(1.0 - dist / r, 0, 1) ** 2
            img_np = img_np + glare_mask[:, :, None] * 180.0

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def get_ssl_transforms(image_size: int = 224):
    """
    Returns (base_aug, hard_aug, highpass_fine_aug, highpass_coarse_aug,
              highpass_legacy_aug)

    base_aug            — easy spatial anchor: mild crop, moderate jitter,
                           mild vignette
    hard_aug            — hard spatial positive: aggressive crop, strong
                           jitter, vignette, glare (NO grayscale/solarize —
                           color is diagnostic in fundus imaging)
    highpass_fine_aug    — fine-scale frequency view (kernel=3):
                           microaneurysm-scale structure
    highpass_coarse_aug  — coarse-scale frequency view (kernel=15):
                           hemorrhage/exudate-scale structure
    highpass_legacy_aug  — single-band (kernel=5), IDENTICAL to the DBFC
                           deepfake setting — used ONLY for the single-band
                           ablation baseline
    """

    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    base_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        IlluminationVignette(strength_range=(0.1, 0.25), p=0.2),
        T.ToTensor(),
        normalize,
    ])

    hard_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        IlluminationVignette(strength_range=(0.2, 0.4), p=0.5),
        GlareSpot(p=0.3),
        T.ToTensor(),
        normalize,
    ])

    highpass_fine_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        HighPassRGB(kernel_size=3),
        T.ToTensor(),
        normalize,
    ])

    highpass_coarse_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        HighPassRGB(kernel_size=15),
        T.ToTensor(),
        normalize,
    ])

    highpass_legacy_aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        HighPassRGB(kernel_size=5),
        T.ToTensor(),
        normalize,
    ])

    return base_aug, hard_aug, highpass_fine_aug, highpass_coarse_aug, highpass_legacy_aug