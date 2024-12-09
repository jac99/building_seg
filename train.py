import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import segmentation_models_pytorch as smp

from dataset.dataset import BuildingSegDataset, get_augmentations


class SegmentationModel(L.LightningModule):
    def __init__(self, base_model: nn.Module, loss: str):
        super().__init__()

        self.model = base_model
        self.loss = loss

        if loss == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss == "focal":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        elif loss == "ce":
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss}")

        # Step metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image: Tensor) -> Tensor:
        mask = self.model(image)
        return mask

    def shared_step(self, batch):
        image = batch["image"]
        # (batch_size, num_channels, height, width)
        assert image.ndim == 4

        mask = batch["mask"]
        assert mask.ndim == 3   # (N, H, W)
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits = self.forward(image)    # (B, C=1, H, W) tensor of float32

        if self.loss == "ce":
            # Convert from (B, H, W) to (B, 1, H, W) format
            mask1 = mask.unsqueeze(1).float()
        else:
            mask1 = mask

        loss = self.loss_fn(logits, mask1)

        # Convert mask values to probabilities and apply thresholding
        prob_mask = logits.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.unsqueeze(1).long(), mode="binary")
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn,"tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {f"{stage}_per_image_iou": per_image_iou, f"{stage}_dataset_iou": dataset_iou}
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch)
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch)
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def configure_optimizers(self):
        lr = 6e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_root = args.dataset
    debug = args.debug

    print(f"Dataset root: {dataset_root}")
    print(f"Debug: {debug}")

    weights_folder = "weights"
    max_size = None
    num_workers = 8
    batch_size = 24
    encoder_weights = "imagenet"

    config = 6

    if debug:
        num_workers = 0
        batch_size = 2
        max_size = 2
        num_epochs = 2
        architecture = "manet"
        encoder_name = "efficientnet-b0"
        loss_fn = "ce"
    else:
        if config == 1:
            num_epochs = 32
            architecture = "Linknet"
            encoder_name = "efficientnet-b1"
            loss_fn = "dice"
        elif config == 2:
            num_epochs = 64
            architecture =  "FPN"
            encoder_name = "efficientnet-b1"
            loss_fn = "ce"
        elif config == 3:
            num_epochs = 64
            architecture = "FPN"
            encoder_name = "efficientnet-b1"
            loss_fn = "ce"
        elif config == 4:
            num_epochs = 96
            architecture = "pspnet"
            encoder_name = "efficientnet-b1"
            loss_fn = "ce"
        elif config == 5:
            num_epochs = 96
            architecture = "manet"
            encoder_name = "efficientnet-b3"
            loss_fn = "ce"
        elif config == 6:
            num_epochs = 196
            architecture = "manet"
            encoder_name = "efficientnet-b3"
            loss_fn = "ce"
        else:
            raise NotImplementedError(f"Incorrect config: {config}")

    transforms = get_augmentations()
    datasets = {
        'train': BuildingSegDataset(dataset_root, mode='train', transform=transforms['train'], max_size=max_size),
        'val': BuildingSegDataset(dataset_root, mode='val', transform=transforms['val'], max_size=max_size)
    }
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    base_model = smp.create_model(architecture, encoder_name=encoder_name, in_channels=3, classes=1,
                                  encoder_weights=encoder_weights)
    #Create PyTorchLightning model wrapper
    model = SegmentationModel(base_model, loss_fn)

    wandb_logger = WandbLogger(project='sem_seg')
    trainer = Trainer(max_epochs=num_epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'])

    #valid_metrics = trainer.validate(model, dataloaders=dataloaders['val'], verbose=False)

    # Save the final model weights
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(weights_folder, exist_ok=True)
    model_filepath = os.path.join("..", weights_folder, f"semseg_{timestamp}")
    model.model.save_pretrained(model_filepath)
    print(f"Final model saved to: {model_filepath}")
