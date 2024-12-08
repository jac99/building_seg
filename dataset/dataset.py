import os
import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torch import Tensor


def denormalize_transform(x: Tensor) -> Tensor:
    inv_trans = v2.Compose([
        v2.Normalize(mean=[0., 0., 0.], std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
        v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    return inv_trans(x)


def visualize(image, mask):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Image")
    img = denormalize_transform(image)
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Mask")
    mask = mask.numpy()
    mask = mask * 255
    plt.imshow(mask)

    plt.show()


def get_augmentations():

    height = 256
    width = 256

    train_transform = [
        A.Resize(height,width),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=180, shift_limit=0.1, p=1, border_mode=0),
        #A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True)
    ]

    val_transform = [
        A.Resize(height,width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True)
    ]

    transforms = {'train': A.Compose(train_transform), 'val': A.Compose(val_transform)}
    return transforms


class BuildingSegDataset(Dataset):
    def __init__(self, dataset_root: str, images_folder='geoportal_orto', masks_folder='geoportal_build_mask',
                 mode: str = "train", transform = None, max_size: int = None):
        self.dataset_root = dataset_root
        self.images_folder = os.path.join(dataset_root, images_folder)
        self.masks_folder = os.path.join(dataset_root, masks_folder)
        self.mode = mode
        self.transform = transform
        self.max_size = max_size
        self.img_ext = '.png'
        self.mask_ext = '.png'

        assert os.path.exists(self.dataset_root), f"Cannot access dataset: {self.dataset_root}"
        assert os.path.exists(self.images_folder), f"Cannot access images: {self.images_folder}"

        # Index images and masks
        images = os.listdir(self.images_folder)
        images.sort()
        images = [e for e in images if os.path.splitext(e)[1] == self.img_ext]

        if self.mode == 'train':
            images = [e for ndx, e in enumerate(images) if ndx % 10 != 0]   # 90% for training
        else:
            images = [e for ndx, e in enumerate(images) if ndx % 10 == 0]   # 10% for annotations

        if self.max_size is not None:
            images = images[:self.max_size]

        self.images = [os.path.join(self.images_folder, e) for e in images]
        mask_names = ["mask_" + e[5:] for e in images]
        self.masks = [os.path.join(self.masks_folder, e) for e in mask_names]
        assert len(self.images) == len(self.masks)
        for mask_file in self.masks:
            assert os.path.exists(mask_file), f"Cannot access mask: {mask_file}"
        print(f"Dataset - mode: {self.mode}   images: {len(self.images)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, ndx: int):
        image_filepath = self.images[ndx]
        mask_filepath = self.masks[ndx]
        assert os.path.exists(image_filepath), f"Cannot access image: {image_filepath}"
        assert os.path.exists(mask_filepath), f"Cannot access mask: {mask_filepath}"

        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert to RGB
        mask = cv2.imread(mask_filepath, flags=cv2.IMREAD_GRAYSCALE)
        # Convert 255 to 1 in a mask (0 denotes a background and 1 denotes foreground (building) class
        mask[mask > 0] = 1

        assert image.ndim == 3
        assert mask.ndim == 2
        assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]
        assert image.shape[2] == 3      # BGR image expected

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample


if __name__ == "__main__":
    # Test the dataset
    dataset_root = "/data/Buildings-geoportal"
    transforms = get_augmentations()
    mode = "train"
    dataset = BuildingSegDataset(dataset_root, mode=mode, transform=transforms[mode])
    print(len(dataset))
    e = dataset[0]
    print(e)
    visualize(e['image'], e['mask'])

