import torch
from vae_model import VAE

def generate_images(model_path, latent_dim=64, num_images=10):
    """
    Generate images using a trained VAE model.

    Parameters:
        model_path (str): Path to the trained VAE model.
        latent_dim (int): The dimension of the latent space.
        num_images (int): The number of images to generate.

    Returns:
        generated_images (Tensor): A tensor containing generated images.
    """
    # Load the trained VAE model
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_images, latent_dim).to(next(vae.parameters()).device)
        # Generate images from the latent vectors
        generated_images = vae.decode(z)
        
    return generated_images
