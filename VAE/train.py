import torch
import os
from torch.utils.data import DataLoader
from vae_model import VAE, vae_loss
from dataset import CustomDataset

def train_vae(data_dir, output_dir, latent_dim=64, batch_size=128, learning_rate=0.001, num_epochs=20):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model, dataset, and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(latent_dim=latent_dim).to(device)
    dataset = CustomDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        vae.train()
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = vae(x)
            loss = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(epoch_loss)
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Adjust learning rate
        scheduler.step()

        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f'vae_epoch_{epoch + 1}.pth')
        torch.save(vae.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved to {checkpoint_path}')

    print('Training completed.')
