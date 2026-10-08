import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from density.csrnet import CSRNet
from utils.dataset import ShanghaiTechDataset
import time

def train():
    # 1. Hyperparameters
    lr = 1e-7
    batch_size = 1
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Model, Loss, Optimizer
    model = CSRNet(load_weights=False).to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4)

    # 3. Data Loaders (Assuming pre-processed .h5 files)
    # Note: Update paths based on Colab environment
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # This is a placeholder path for the user to update in Colab
    train_dataset = ShanghaiTechDataset(
        img_root='data/shanghaitech/part_A/train_data/images',
        gt_root='data/shanghaitech/part_A/train_data/ground-truth-h5',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)
            
            output = model(img)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/csrnet/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
