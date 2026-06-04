import os
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from src.dataset import smiles_to_rich_graph  # Memanggil dataset.py bawaan repo Anda

class ClinToxDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'data/', batch_size: int = 128, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_graphs, self.val_graphs, self.test_graphs = None, None, None

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            val_df = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))
            self.train_graphs = [smiles_to_rich_graph(s, y) for s, y in zip(train_df['Drug'], train_df['Y'])]
            self.val_graphs = [smiles_to_rich_graph(s, y) for s, y in zip(val_df['Drug'], val_df['Y'])]
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            self.test_graphs = [smiles_to_rich_graph(s, y) for s, y in zip(test_df['Drug'], test_df['Y'])]

    def train_dataloader(self):
        return DataLoader(self.train_graphs, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_graphs, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_graphs, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
