import torch
from torch_geometric.data import Data
from rdkit import Chem

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

def smiles_to_rich_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    node_features = [get_atom_features_rich(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    
    return Data(x=x, edge_index=edge_index, y=y)
