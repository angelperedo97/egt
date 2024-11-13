import warnings

# Suppress warnings to make the output cleaner. Update libraries to avoid this
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import dgl
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from model import GraphTransformerNet
from torch.utils.data import DataLoader
from data_loading import LigandDataset, get_ligand_files, get_atom_types, split_ligand_files, collate_fn

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup dataset and dataloader
root_directory = 'CrossDocked2020'
ligand_files = get_ligand_files(root_directory)
atom_types_list = get_atom_types(ligand_files)
atom_type_to_idx = {atom_type: idx for idx, atom_type in enumerate(atom_types_list)}

# Split files into training and test sets
train_ligand_files, test_ligand_files = split_ligand_files(ligand_files)
train_dataset = LigandDataset(ligand_files=train_ligand_files, atom_type_to_idx=atom_type_to_idx)
test_dataset = LigandDataset(ligand_files=test_ligand_files, atom_type_to_idx=atom_type_to_idx)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

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

# Initialize model and optimizer, and move the model to GPU
model = GraphTransformerNet(net_params).to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

# Initialize lists for tracking metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0
    for batched_graph, labels in train_loader:
        labels = labels.float().to(device)  # For BCEWithLogitsLoss, labels need to be float

        # Move batched_graph to GPU
        batched_graph = batched_graph.to(device)

        # Forward pass
        h = batched_graph.ndata['feat']
        scores = model(batched_graph, h)  # Output shape: [batch_size]

        # Compute loss
        loss = model.loss(scores, labels)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert logits to probabilities and binary predictions
        probabilities = torch.sigmoid(scores)  # Shape: [batch_size]
        predictions = (probabilities > 0.5).float()  # Shape: [batch_size]

        # Compute accuracy
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

    # Calculate average training loss and accuracy for the epoch
    average_loss = epoch_loss / len(train_loader)
    accuracy = total_correct / total_samples
    train_losses.append(average_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.4f}")
    
    # print some predictions and actual labels for inspection
    # print("Sample Predictions:", predictions[:5].cpu().numpy())
    # print("Sample Actual Labels:", labels[:5].cpu().numpy())
    
    # Testing phase at the end of each epoch
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total_samples = 0
    
    with torch.no_grad():
        for batched_graph, labels in test_loader:
            labels = labels.float().to(device)
            batched_graph = batched_graph.to(device)

            # Forward pass for testing
            h = batched_graph.ndata['feat']
            scores = model(batched_graph, h)
            loss = model.loss(scores, labels)
            test_loss += loss.item()

            # Compute test predictions and accuracy
            probabilities = torch.sigmoid(scores)
            predictions = (probabilities > 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total_samples += labels.size(0)

    # Calculate average testing loss and accuracy
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total_samples
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plotting the training and testing metrics
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 8))

# Training and test loss
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss")
plt.legend()

# Training and test accuracy
plt.subplot(2, 1, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_testing_metrics.png")
plt.show()







