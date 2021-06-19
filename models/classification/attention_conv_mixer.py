import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from typing import Union, Type

from modules import AttentionConvMixer, Conv2d


class AttentionConvMixerClassifier(pl.LightningModule):

    def __init__(
        self, 
        input_size: Union[int, tuple],
        in_channels: int,
        out_channels: int,
        feedforward_dim: int,
        patch_size: Union[int, tuple],
        n_attention_layers: int,
        n_heads: int,
        n_classes: int,
        kernel_size: int,
        dropout: float=0.0,
        activation_conv: str = 'relu',
        activation_attn: str = 'GELU',
        conv_type: Type = Conv2d
    ):
        super().__init__()
        self.n_classes = n_classes

        self.backbone = AttentionConvMixer(
            n_attention_layers=n_attention_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            kernel_size=kernel_size,
            feedforward_dim=feedforward_dim,
            patch_size=patch_size,
            n_heads=n_heads,
            dropout=dropout,
            activation_conv=activation_conv,
            activation_attn=activation_attn,
            dropout=dropout,
            conv_type=conv_type
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten()
        )

        self.classifier = nn.Linear(out_channels, n_classes)

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
