import torch
import os
from torch.utils.data import DataLoader
from dataset import CustomDataset

def train_(model, dataloader, epochs):
  
    for epoch in range(epochs):
        model.generator.train()
        model.discriminator.train()
    
        d_loss_total = 0.0
        g_loss_total = 0.0
    
        for real_images in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            real_images = real_images.to(device)  
            losses = model.train_step(real_images)
            d_loss_total += losses['d_loss']
            g_loss_total += losses['g_loss']
    
        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss_total / len(dataloader)}, G Loss: {g_loss_total / len(dataloader)}")
    
    torch.save(dcgan.generator.state_dict(), 'generator.pth')
    torch.save(dcgan.discriminator.state_dict(), 'discriminator.pth')

    return model
