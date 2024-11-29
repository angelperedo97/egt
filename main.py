import warnings

# Suppress warnings to make the output cleaner. Update libraries to avoid this
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import AffinityPredictionModel
from data_loading import AffinityDataset, load_affinity_data, split_affinity_data, collate_fn


# Prepare data
rmsd_threshold = 2  # Filter poor poses by RMSD to the crystal pose
df = load_affinity_data('receptor_ligand_pairs.pkl', rmsd_threshold)
train_df, test_df = split_affinity_data(df, test_size=0.2, random_state=42)

# Define the atom_type_to_idx mapping
atom_types = [
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Mg',
    'Mn', 'Zn', 'Ca', 'Fe', 'B'
]
atom_type_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}

# Define the base directory where 'CrossDocked2020' resides
base_dir = os.getcwd() + '/CrossDocked2020/'

# Define the dataset
train_dataset = AffinityDataset(train_df, atom_type_to_idx, base_dir=base_dir)
test_dataset = AffinityDataset(test_df, atom_type_to_idx, base_dir=base_dir)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define network parameters
net_params = {
    'hidden_dim': 160,
    'n_heads': 8,
    'out_dim': 160,
    'in_feat_dropout': 0.0,
    'dropout': 0.0,
    'L': 12,  # Number of layers   
    'layer_norm': False,
    'batch_norm': True,
    'residual': True,
    'num_atom_type': len(atom_type_to_idx)
}

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AffinityPredictionModel(net_params).to(device)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# Lists to store losses for each epoch
train_losses = []
val_losses = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    # Create a tqdm progress bar for the training loop
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batched_receptor_graph, batched_ligand_graph, labels in pbar:
            batched_receptor_graph = batched_receptor_graph.to(device)
            batched_ligand_graph = batched_ligand_graph.to(device)
            labels = labels.to(device)

            # Get node features
            receptor_features = batched_receptor_graph.ndata['feat']
            ligand_features = batched_ligand_graph.ndata['feat']

            # Forward pass
            outputs = model(batched_receptor_graph, receptor_features,
                           batched_ligand_graph, ligand_features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar description with current loss
            pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store train loss for plotting

    # Evaluation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batched_receptor_graph, batched_ligand_graph, labels in test_loader:
            batched_receptor_graph = batched_receptor_graph.to(device)
            batched_ligand_graph = batched_ligand_graph.to(device)
            labels = labels.to(device)

            receptor_features = batched_receptor_graph.ndata['feat']
            ligand_features = batched_ligand_graph.ndata['feat']

            outputs = model(batched_receptor_graph, receptor_features,
                           batched_ligand_graph, ligand_features)

            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)  # Store validation loss for plotting

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# After training is done, plot the losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss (MSE)', color='b', linestyle='-', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss (MSE)', color='r', linestyle='-', marker='x')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('training_validation_loss.png')









