import os
import dgl
import torch
import pickle
import random
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split


def get_ligand_files(root_dir, cache_file='ligand_files_cache.pkl', use_cache=True):
    """
    Retrieves a list of ligand file paths from the specified root directory.
    If a cache file exists, it loads the cached list. Otherwise, it scans the
    directory for ligand files (SDF files) and saves the list to a cache file
    for future use.
    """
    if use_cache and os.path.exists(cache_file):
        # Load the cached list of ligand file paths
        with open(cache_file, 'rb') as f:
            ligand_files = pickle.load(f)
    else:
        # Traverse directories to find ligand files
        ligand_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.sdf'):
                    ligand_files.append(os.path.join(dirpath, filename))
        # Save the list to a cache file for future use
        with open(cache_file, 'wb') as f:
            pickle.dump(ligand_files, f)
        print("Ligand file paths cached for future use.")
    return ligand_files
    
    
def get_atom_types(ligand_files, atom_types_file='atom_types_set.pkl', use_cache=True):
    """
    Retrieves a set of unique atom types from the specified ligand files.
    If a cache file exists, it loads the cached set. Otherwise, it scans the
    ligand files for atom types (using RDKit), saves the set to a cache file,
    and returns the set.
    """
    if use_cache and os.path.exists(atom_types_file):
        # Load the atom types set from the file
        with open(atom_types_file, 'rb') as f:
            atom_types_set = pickle.load(f)
    else:
        # Initialize a set to collect unique atom types
        atom_types_set = set()
        
        # Scan each ligand file to collect atom types
        print("Scanning ligand files to collect atom types...")
        
        # Use tqdm for a progress bar
        for file_path in tqdm(ligand_files, desc="Processing ligands"):
            try:
                # Read the molecule(s) from the SDF file
                supplier = Chem.SDMolSupplier(file_path, removeHs=False)
                for mol in supplier:
                    if mol is None:
                        print(f"Warning: Failed to read molecule from {file_path}")
                        continue
                    try:
                        Chem.SanitizeMol(mol)
                    except Chem.SanitizeException as e:
                        print(f"Sanitization failed for molecule in {file_path}: {e}")
                        continue  # Skip this molecule
                    for atom in mol.GetAtoms():
                        element = atom.GetSymbol().capitalize()
                        if element:
                            atom_types_set.add(element)
                        else:
                            atom_types_set.add('Unknown')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save the atom types set to a file for future use
        with open(atom_types_file, 'wb') as f:
            pickle.dump(atom_types_set, f)
        print("\nUnique atom types found in the dataset:")
        print(sorted(atom_types_set))
        print(f"Atom types set saved to '{atom_types_file}'.")
    
    return atom_types_set
    
    
# Split the ligand files into training and testing subsets
def split_ligand_files(ligand_files, test_size=0.2, random_state=42):
    train_files, test_files = train_test_split(ligand_files, test_size=test_size, random_state=random_state)
    return train_files, test_files
    

# Has both molecular graphs and synthetic graphs
class LigandDataset(Dataset):
    def __init__(self, ligand_files, atom_type_to_idx, transform=None, pre_transform=None):
        super(LigandDataset, self).__init__(None, transform, pre_transform)
        self.ligand_files = ligand_files
        self.atom_type_to_idx = atom_type_to_idx
        self.total_len = len(self.ligand_files) * 2  # Half real, half fake

    def len(self):
        return self.total_len

    def get(self, idx):
        if idx % 2 == 0 and idx // 2 < len(self.ligand_files):  
            # If index is even, get a real molecule if available
            data = self.load_real_graph(idx // 2)
            data.y = torch.tensor(1, dtype=torch.float)  # Label 1 for real graphs
        else:
            # If index is odd or we've exhausted real molecules, get a synthetic graph
            data = self.generate_synthetic_graph()
            data.y = torch.tensor(0, dtype=torch.float)  # Label 0 for synthetic graphs

        return data

    def load_real_graph(self, idx):
        file_path = self.ligand_files[idx]
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)
        mol = supplier[0]
        if mol is None:
            raise ValueError(f"Failed to read molecule from {file_path}")

        # Sanitize the molecule
        try:
            Chem.SanitizeMol(mol)
        except Chem.SanitizeException as e:
            raise ValueError(f"Sanitization failed for molecule in {file_path}: {e}")

        # Extract atom type indices
        atom_indices = []
        for atom in mol.GetAtoms():
            element = atom.GetSymbol().capitalize()
            atom_type_idx = self.atom_type_to_idx.get(element, self.atom_type_to_idx.get('Unknown'))
            atom_indices.append(atom_type_idx)

        x = torch.tensor(atom_indices, dtype=torch.long)

        # Extract edge indices
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected edges

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)

        return data

    def generate_synthetic_graph(self):
        num_nodes = random.randint(5, 20)
        num_features = len(self.atom_type_to_idx)
        x = torch.randint(0, num_features, (num_nodes,), dtype=torch.long)

        edge_index = []
        for _ in range(random.randint(5, num_nodes * 2)):
            i, j = random.sample(range(num_nodes), 2)
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected edges

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)

        return data

        
        
"""
Since the dataset uses torch_geometric.data.Data, but the model is built on DGL (dgl.DGLGraph), 
some tweaks are needed to ensure compatibility between PyTorch Geometric data and DGL
"""
def pyg_to_dgl(data):
    # Convert PyTorch Geometric data to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)
    
    # Convert NetworkX graph to DGL graph
    dgl_graph = dgl.from_networkx(nx_graph)
    
    # Add node features
    dgl_graph.ndata['feat'] = data.x.type(torch.long)  # Ensure it's torch.long
    
    # Add labels
    dgl_graph.y = data.y
    
    return dgl_graph
    

def collate_fn(batch):
    dgl_graphs = [pyg_to_dgl(data) for data in batch]
    labels = torch.stack([graph.y for graph in dgl_graphs])  # Stack labels into a tensor
    batched_graph = dgl.batch(dgl_graphs)  # Batch graphs for DGL
    return batched_graph, labels



