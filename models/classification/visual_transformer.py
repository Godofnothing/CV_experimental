import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from typing import Union

from modules import VisualTransformer


class ViTClassifier(pl.LightningModule):
    supported_classification_modes = ['token', 'pool']

    def __init__(
        self, 
        input_size: Union[int, tuple],
        hidden_dim: int, 
        patch_size: Union[int, tuple],
        n_layers: int,
        n_heads: int,
        n_classes: int,
        dropout: float=0.0,
        activation: str = 'GELU',
        classification_mode: str = 'token'
    ):
        super().__init__()
        self.n_classes = n_classes
        self.classification_mode = classification_mode

        assert classification_mode in self.supported_classification_modes

        self.backbone = VisualTransformer(
            input_size=input_size,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            create_cls_token=(classification_mode == 'token')
        )

        if classification_mode:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten()
            )

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, images : torch.Tensor):
        embeddings = self.backbone(images)

        if self.classification_mode == 'pool':
            embeddings = self.pool(embeddings)
        else:
            # get the CLS token
            embeddings = embeddings[0]

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
