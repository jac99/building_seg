import argparse
import os
import numpy as np
import tqdm
import cv2
import torch
import torch.nn as nn
from torch import Tensor
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.dataset import filter_images


DEBUG = False


def preprocess_image(image) -> Tensor:
    transform = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    sample = transform(image=image)
    return sample['image']


def process_image(image: str, model: nn.Module, device, output_path: str):
    assert os.path.exists(image), f"Cannot access image: {image}"
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert to RGB
    x = preprocess_image(image)
    x = x.to(device)
    with torch.no_grad():
        mask = model(x.unsqueeze(0)).squeeze(0).cpu()
    # mask is (C=1, H, W) tensor of logits
    threshold = 0.5
    mask = ((mask.sigmoid() > threshold).permute(1,2,0) * 255).numpy().astype(np.uint8)
    # mask is (H, W, C=1) ndarray of uint8
    # Make the segmentation results blue
    temp = np.zeros_like(mask)
    mask = np.concatenate([mask, temp, temp], axis=2)     # (H, W, C=3)
    image_name = os.path.split(image)[1]
    out_filepath =os.path.join(output_path, "mask_" + image_name)
    cv2.imwrite(out_filepath, mask)


def process_images(images: list[str], model: nn.Module, device):
    print(f"Processing video...")
    output_path = 'output'
    os.makedirs(output_path, exist_ok=True)

    for image in tqdm.tqdm(images):
        process_image(image, model, device, output_path )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=['train', 'val'])

    args = parser.parse_args()

    assert os.path.exists(args.split), f"Cannot access images folder: {args.image_path}"
    assert os.path.exists(args.checkpoint), f"Cannot access model checkpoint: {args.checkpoint}"

    print(f"Images folder: {args.image_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # Index images and masks
    images = os.listdir(args.image_path)
    images = [e for e in images if os.path.splitext(e)[1] == '.png']
    images = filter_images(images, args.split)
    print(f"{len(images)} images in {args.split} split")

    # Instantiate the model and load weights
    model = smp.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    process_images(images, model, device)
