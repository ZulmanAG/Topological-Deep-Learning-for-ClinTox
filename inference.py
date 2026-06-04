import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from rdkit import Chem

# 1. DEFINISI ARSITEKTUR MODEL (Wajib Sama dengan Training)
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

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_combined = torch.cat([x_mean, x_max], dim=1) 
        return self.mlp(x_combined)

# 2. FITUR EKSTRAKSI DENGAN PENANGANAN ERROR (SAFE FEATURING)
def get_atom_features_rich(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge(),      
        atom.GetTotalNumHs(),        
        atom.GetNumRadicalElectrons()
    ]

def smiles_to_rich_graph_safe(smiles):
    try:
        if not isinstance(smiles, str) or smiles.strip() == "":
            raise ValueError("SMILES kosong")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit gagal membaca SMILES")
        node_features = [get_atom_features_rich(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(node_features, dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(i, j), (j, i)])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index), False
    except Exception as e:
        x_dummy = torch.zeros((1, 7), dtype=torch.float)
        edge_dummy = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x_dummy, edge_index=edge_dummy), True

# 3. PIPELINE INFERENSI UTAMA
def run_inference(input_path, output_path, checkpoint_path, smiles_column='Drug'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Menjalankan inferensi menggunakan device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint tidak ditemukan di: {checkpoint_path}")
    
    print(f"[INFO] Memuat bobot model dari checkpoint...")
    model = GATv2ToxicityPredictor.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] File input tidak ditemukan di: {input_path}")
        
    df_test = pd.read_csv(input_path)
    print(f"[INFO] Berhasil membaca {len(df_test)} baris.")

    graphs = []
    error_indices = []
    for idx, row in df_test.iterrows():
        smiles = row[smiles_column]
        graph, is_error = smiles_to_rich_graph_safe(smiles)
        graphs.append(graph)
        if is_error:
            error_indices.append(idx)
            
    if len(error_indices) > 0:
        print(f"[WARNING] Ditemukan {len(error_indices)} SMILES rusak di baris: {error_indices}")

    loader = DataLoader(graphs, batch_size=128, shuffle=False)
    predictions = []
    print("[INFO] Memulai kalkulasi prediksi...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())

    for idx in error_indices:
        predictions[idx] = 0.5000

    df_output = pd.DataFrame({
        'Drug_ID': df_test['Drug_ID'] if 'Drug_ID' in df_test.columns else df_test.index,
        'Drug': df_test[smiles_column],
        'Prediction': predictions
    })
    df_output.to_csv(output_path, index=False)
    print(f"[SUCCESS] Prediksi disimpan ke {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--smiles_col", type=str, default="Drug")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.checkpoint, args.smiles_col)
