import os
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# Impor lokal dari folder src dan data
from src.dataset import smiles_to_rich_graph
from src.models import GATv2ToxicityPredictor

def main():
    pl.seed_everything(42, workers=True)
    
    # Proteksi: Unduh data otomatis jika belum ada di folder data/
    if not os.path.exists("data/train.csv"):
        from data.get_data import download_and_split
        download_and_split()
        
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/valid.csv")
    
    print("[INFO] Mengekstrak struktur graf...")
    train_graphs = [g for _, row in train_df.iterrows() if (g := smiles_to_rich_graph(row['Drug'], row['Y'])) is not None]
    val_graphs = [g for _, row in val_df.iterrows() if (g := smiles_to_rich_graph(row['Drug'], row['Y'])) is not None]
    
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False, num_workers=2)
    
    model = GATv2ToxicityPredictor(node_in_dim=7, hidden_dim=128, heads=4, lr=1e-3)
    
    checkpoint = ModelCheckpoint(monitor='val_auroc', mode='max', save_top_k=1, dirpath='gat_models/')
    early_stop = EarlyStopping(monitor='val_auroc', patience=15, mode='max')
    
    if not getattr(wandb.api, 'api_key', None):
        logger = WandbLogger(project='Toxicity_Prediction', name='GATv2_Modular', mode='disabled')
    else:
        logger = WandbLogger(project='Toxicity_Prediction', name='GATv2_Modular')
        
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint, early_stop]
    )
    
    print("[INFO] Memulai pelatihan modular...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
