import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torchmetrics import AUROC, AveragePrecision

class GATv2ToxicityPredictor(pl.LightningModule):
    def __init__(self, node_in_dim=7, hidden_dim=128, heads=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads)
        
        mlp_input_dim = hidden_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_prauc = AveragePrecision(task="binary")

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_combined = torch.cat([x_mean, x_max], dim=1)
        
        return self.mlp(x_combined)

    def training_step(self, data, batch_idx):
        logits = self(data.x, data.edge_index, data.batch).squeeze(-1)
        loss = self.criterion(logits, data.y)
        self.train_auroc(logits, data.y.int())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data.x, data.edge_index, data.batch).squeeze(-1)
        loss = self.criterion(logits, data.y)
        self.val_auroc(logits, data.y.int())
        self.val_prauc(logits, data.y.int())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_prauc', self.val_prauc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
            }
        }
