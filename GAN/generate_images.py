import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def generate(model, num_images = 8, latent_dim = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(16, 2))

    dcgan.generator.to(device)
    dcgan.generator.eval()

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)

        noise = torch.randn(1, latent_dim, device=device)

        with torch.no_grad():
            generated_image = dcgan.generator(noise)

        generated_image = generated_image.squeeze().cpu().detach()
        generated_image = (generated_image*1) * 255

        image = Image.fromarray(np.uint8(generated_image.permute(1, 2, 0).numpy()))

        plt.imshow(image)
        plt.axis('off')

    plt.show()
