import os
import dgl
import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split


def load_affinity_data(pkl_path, rmsd_threshold):
    """
    Loads affinity data from a pickle file and filters out rows where RMSD is greater than
    the specified threshold.

    Args:
        pkl_path (str): Path to the pickle file.
        rmsd_threshold (float): Threshold value for filtering rows based on RMSD.

    Returns:
        pd.DataFrame: Filtered DataFrame containing affinity data.
    """
    df = pd.read_pickle(pkl_path)
    # Ensure the columns are named correctly
    required_columns = ['rmsd', 'vina_score', 'clipped_receptor_path', 'ligand_path']
    assert all(col in df.columns for col in required_columns), \
        f"DataFrame must contain columns: {required_columns}"
    
    # Filter the DataFrame based on the RMSD threshold
    df = df[df['rmsd'] <= rmsd_threshold]
    
    return df


def split_affinity_data(df, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        test_size (float, optional): Proportion of data to use for the test set. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Training and test DataFrames.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df
    
    
def parse_pdb(file_path):
    """Parse a PDB file and return a list of atoms with their coordinates."""
    atoms = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_type = line[76:78].strip()  # Element symbol (columns 77-78)
                x = float(line[30:38])          # X coordinate (columns 31-38)
                y = float(line[38:46])          # Y coordinate (columns 39-46)
                z = float(line[46:54])          # Z coordinate (columns 47-54)
                atoms.append((atom_type, np.array([x, y, z])))
    return atoms
    

def infer_bonds(pdb_path):
    """Infer bonds based on atom proximity and bond thresholds."""
    # Constants
    COVALENT_RADII = {
        'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'P': 1.07,
        'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
        'Mg': 1.41, 'Mn': 1.39, 'Zn': 1.20, 'Ca': 1.76,
        'Fe': 1.32, 'B': 0.87
    }

    MAX_BONDS = {
        'C': 4, 'N': 4, 'O': 2, 'S': 6, 'P': 5,
        'F': 1, 'Cl': 3, 'Br': 3, 'I': 3,
        'Mg': 2, 'Mn': 7, 'Zn': 2, 'Ca': 2,
        'Fe': 6, 'B': 3
    }

    THRESHOLD_TOLERANCE = 0.2  # Å (tolerance for bond distance)

    try:
        atoms = parse_pdb(pdb_path)
        if not atoms:
            # print(f"Warning: No atoms found in PDB file '{pdb_path}'. Skipping.")
            return None, None  # Return None for bond list and atoms

        coords = np.array([coord for _, coord in atoms])

        # Ensure coords is a 2D array with shape (N, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            # print(f"Warning: Invalid coordinates shape in file '{pdb_path}': {coords.shape}. Skipping.")
            return None, None  # Return None for bond list and atoms

        tree = cKDTree(coords)

        bonds = {i: [] for i in range(len(atoms))}

        # Step 1: Iterate through each atom to form bonds
        for i, (atom1, coord1) in enumerate(atoms):
            radius1 = COVALENT_RADII.get(atom1, None)
            max_bonds1 = MAX_BONDS.get(atom1, 0)
            if radius1 is None:
                # print(f"Warning: Unknown atom type '{atom1}' encountered in file '{pdb_path}'. Skipping.")
                continue

            # Step 2: Find neighboring atoms within the bonding radius
            neighbors = tree.query_ball_point(coord1, radius1 + max(COVALENT_RADII.values()) + THRESHOLD_TOLERANCE)

            # Step 3: Attempt to bond with closest atoms within the radius
            for j in neighbors:
                if i >= j:  # Skip bonds to self or duplicate bonds
                    continue
                atom2, coord2 = atoms[j]
                radius2 = COVALENT_RADII.get(atom2, None)
                max_bonds2 = MAX_BONDS.get(atom2, 0)
                if radius2 is None:
                    # print(f"Warning: Unknown atom type '{atom2}' encountered in file '{pdb_path}'. Skipping.")
                    continue

                # Calculate bond threshold (sum of covalent radii + THRESHOLD_TOLERANCE)
                bond_threshold = radius1 + radius2 + THRESHOLD_TOLERANCE

                # Only connect atoms that are within the bonding threshold
                if np.linalg.norm(coord1 - coord2) <= bond_threshold:
                    # Ensure that atoms are not overbonded
                    if len(bonds[i]) < max_bonds1 and len(bonds[j]) < max_bonds2:
                        bonds[i].append(j)
                        bonds[j].append(i)

        # Step 4: Convert bonds into a more readable format (pairs of atom indices)
        bond_list = []
        for i, connected_atoms in bonds.items():
            for j in connected_atoms:
                if i < j:  # Avoid duplicate bonds
                    bond_list.append((i, j))

        return bond_list, atoms

    except Exception as e:
        # print(f"Error processing file '{pdb_path}': {e}")
        return None, None  # Return None for bond list and atoms

    
    
def mol_to_graph_data_obj(bond_list, atoms, atom_type_to_idx):
    """
    Converts inferred bond data to a PyTorch Geometric Data object.
    
    Args:
        bond_list (list): List of bonds as tuples of atom indices (i, j).
        atoms (list): List of atoms with their coordinates and types.
        atom_type_to_idx (dict): Mapping from element symbols to indices.
        
    Returns:
        torch_geometric.data.Data or None: Graph data object, or None if the molecule is invalid.
    """
    if not bond_list or not atoms:
        # print("Invalid bond list or atom list. Skipping processing.")
        return None

    # Atom features (mapping element symbols to indices)
    atom_features = []
    for atom, _ in atoms:
        if atom not in atom_type_to_idx:
            raise ValueError(f"Unknown atom type '{atom}' encountered.")
        atom_features.append(atom_type_to_idx[atom])
    
    x = torch.tensor(atom_features, dtype=torch.long)

    # Edge indices (from bond list)
    edge_index = []
    for i, j in bond_list:
        edge_index.append([i, j])
        edge_index.append([j, i])  # For undirected graph
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data
    
    
class AffinityDataset(Dataset):
    def __init__(self, dataframe, atom_type_to_idx, transform=None, pre_transform=None, base_dir=''):
        """
        Initializes the dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing molecular data.
            atom_type_to_idx (dict): Mapping from element symbols to indices.
            transform (callable, optional): A function/transform to apply to each data sample.
            pre_transform (callable, optional): A function/transform to apply to data before saving to disk.
            base_dir (str, optional): Base directory for file paths. Defaults to ''.
        """
        super(AffinityDataset, self).__init__(None, transform, pre_transform)
        self.dataframe = dataframe
        self.atom_type_to_idx = atom_type_to_idx
        self.base_dir = base_dir  # Default to empty string if base_dir is not provided

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        vina_score = row['vina_score']
        receptor_path = os.path.join(self.base_dir, row['clipped_receptor_path'])
        ligand_path = os.path.join(self.base_dir, row['ligand_path'])

        # Process receptor
        receptor_bonds, receptor_atoms = infer_bonds(receptor_path)
        receptor_data = mol_to_graph_data_obj(receptor_bonds, receptor_atoms, self.atom_type_to_idx)  # PyTorch Geometric Data object
        if receptor_data is None:
            # print(f"Failed to process receptor at index {idx}. Skipping sample.")
            return None, None, None

        # Process ligand
        ligand_bonds, ligand_atoms = infer_bonds(ligand_path)
        ligand_data = mol_to_graph_data_obj(ligand_bonds, ligand_atoms, self.atom_type_to_idx) # PyTorch Geometric Data object
        if ligand_data is None:
            # print(f"Failed to process ligand at index {idx}. Skipping sample.")
            return None, None, None

        return receptor_data, ligand_data, torch.tensor(vina_score, dtype=torch.float)


def pyg_to_dgl(data):
    """
    Converts a PyTorch Geometric Data object to a DGLGraph.

    Args:
        data (torch_geometric.data.Data): The PyG data object.

    Returns:
        dgl.DGLGraph: The converted DGL graph.
    """
    # Convert to NetworkX graph
    nx_graph = to_networkx(data, to_undirected=True)

    # Create DGL graph from NetworkX graph
    dgl_graph = dgl.from_networkx(nx_graph)

    # Add node features
    # Convert node features to long tensor
    dgl_graph.ndata['feat'] = data.x.type(torch.long)

    return dgl_graph


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle receptor and ligand graphs.

    Args:
        batch (list of tuples): Each tuple contains (receptor_data, ligand_data, vina_score).

    Returns:
        tuple: Batched receptor graphs, batched ligand graphs, and labels.
    """
    receptor_graphs = []
    ligand_graphs = []
    labels = []
    skipped_samples = 0

    for receptor_data, ligand_data, vina_score in batch:
        if receptor_data is None or ligand_data is None:
            skipped_samples += 1
            continue  # Skip this sample

        receptor_graph = pyg_to_dgl(receptor_data)
        ligand_graph = pyg_to_dgl(ligand_data)

        receptor_graphs.append(receptor_graph)
        ligand_graphs.append(ligand_graph)
        labels.append(vina_score)

    # if skipped_samples > 0:
        # print(f"Skipped {skipped_samples} samples due to processing failures.")

    if len(receptor_graphs) == 0 or len(ligand_graphs) == 0:
        # print("Warning: All samples in the batch failed to process. Skipping this batch.")
        return None, None, None  # Return None to indicate batch failure

    # Batch the receptor and ligand graphs separately
    batched_receptor_graph = dgl.batch(receptor_graphs)
    batched_ligand_graph = dgl.batch(ligand_graphs)

    # Stack labels into a tensor
    labels = torch.stack(labels)  # Shape: [batch_size]

    return batched_receptor_graph, batched_ligand_graph, labels

