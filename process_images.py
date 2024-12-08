import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch import Tensor
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEBUG = False


def preprocess_frame(frame) -> Tensor:
    transform = A.Compose([
        A.PadIfNeeded(256, 256, position="top_left"),       # Make it multiple of 32
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    sample = transform(image=frame)
    return sample['image']


def process_images(video_filepath: str, model: nn.Module, device):
    print(f"Processing video...")
    masked_filepath = os.path.splitext(args.video)[0] + "_mask.mp4"
    blended_filepath = os.path.splitext(args.video)[0] + "_blended.mp4"

    cap = cv2.VideoCapture(video_filepath)
    video_writer = cv2.VideoWriter_fourcc(*'XVID')
    masked_out_stream = None
    blended_out_stream = None

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        x = preprocess_frame(frame)
        x = x.to(device)
        with torch.no_grad():
            mask = model(x.unsqueeze(0)).squeeze(0).cpu()
        # mask is (C=1, H, W) tensor of logits
        mask = ((mask.sigmoid() > 0.5).permute(1,2,0) * 255).numpy().astype(np.uint8)
        # mask is (H, W, C=1) ndarray of uint8
        # Get to the original frame size
        mask = mask[:frame.shape[0], :frame.shape[1], :]
        # Make the segmentation results blue
        temp = np.zeros_like(mask)
        mask = np.concatenate([mask, temp, temp], axis=2)     # (H, W, C=3)

        frame = cv2.addWeighted(frame, .4, mask, 1., 0.)
        if masked_out_stream is None:
            w, h = frame.shape[1], frame.shape[0]
            # Last is isColor flag
            masked_out_stream = cv2.VideoWriter(masked_filepath, video_writer, 30., (w, h), True)
            blended_out_stream = cv2.VideoWriter(blended_filepath, video_writer, 30., (w, h), True)

        assert frame.shape[0] == h and frame.shape[1] == w
        assert mask.shape[0] == h and mask.shape[1] == w

        masked_out_stream.write(mask)
        blended_out_stream.write(frame)
        count = count + 1
        if DEBUG and count > 30:
            break

    cap.release()
    masked_out_stream.release()
    blended_out_stream.release()
    print("Finito")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.video), f"Cannot access images folder: {args.video}"
    assert os.path.exists(args.checkpoint), f"Cannot access model checkpoint: {args.checkpoint}"

    print(f"Images folder: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")

    # Instantiate the model and load weights
    model = smp.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    process_images(args.video, model, device)
