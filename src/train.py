import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from src.model import UAVDetector

class UAVAudioDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = []
        self.labels = []
        self.class_map = {"mavic1": 0, "mavic2": 1, "mini": 2}
        
        for class_name, label_idx in self.class_map.items():
            folder_path = os.path.join(data_dir, class_name)
            if os.path.exists(folder_path):
                files = glob.glob(os.path.join(folder_path, "*.pt"))
                self.file_paths.extend(files)
                self.labels.extend([label_idx] * len(files))
        
        if len(self.file_paths) > 0:
            sample_tensor = torch.load(self.file_paths[0], weights_only=True)
            self.signal_length = sample_tensor.shape[0]
        else:
            self.signal_length = 0
            
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tensor_path = self.file_paths[idx]
        audio_chunk = torch.load(tensor_path, weights_only=True)
        audio_chunk = audio_chunk.unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_chunk, label

def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.001):
    print("Initializing Dataset and DataLoader...")
    dataset = UAVAudioDataset(data_dir)
    
    if len(dataset) == 0:
        print("No data found!")
        return None
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset Split: {train_size} Training | {val_size} Validation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVDetector(signal_length=dataset.signal_length, m_coeffs=20, device=device).to(device)
    
    class_weights = torch.tensor([1.0, 1.3, 1.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"Training on device: {device}...")

    # Create our CSV tracking file and write the headers
    with open('training_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {running_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss/len(val_loader):.4f} "
              f"| Val Acc: {val_acc:.2f}%")
        
        # Log the metrics to the CSV file
        with open('training_history.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, running_loss/len(train_loader), train_acc, avg_val_loss, val_acc, current_lr])

    print("Training Complete!")
    return model