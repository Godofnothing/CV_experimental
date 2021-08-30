import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from modules import ResNet

class ResNetClassifier(pl.LightningModule):

    def __init__(
        self, 
        config : dict,
        n_classes: int,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.backbone = ResNet(config=config)
        output_channels = self.backbone.get_output_channels()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten()
        )

        self.classifier = nn.Linear(output_channels, n_classes)

    def forward(self, images : torch.Tensor):
        embeddings = self.backbone(images)
        embeddings = self.pool(embeddings)
        logits = self.classifier(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train/loss', loss)

        pred_labels = logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, labels)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log('val/loss', loss)

        pred_labels = logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, labels)
        self.log('val/accuracy', acc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self.forward(images)
        pred_labels = logits.argmax(dim=-1)
        return {'preds': pred_labels, 'labels': labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([o['preds'] for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        acc = torchmetrics.functional.accuracy(preds, labels)

        self.log('test/accuracy', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "val/accuracy"}
