import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_images(model, num_samples, mean, std):
    random_vectors = torch.normal(mean=mean_embedding.expand(num_samples, -1, -1, -1), 
                               std=std_embedding.expand(num_samples, -1, -1, -1))*1.2
    
    with torch.no_grad():
        decoded_images = model.decoder(random_vectors.cuda())  



    num_images = decoded_images.shape[0]

    plt.figure(figsize=(num_images * 2, 2))  

    for i in range(num_images):
        image = decoded_images[i].permute(1, 2, 0).cpu().numpy() 
        
        plt.subplot(1, num_images, i + 1)  
        plt.imshow(image)
        plt.axis('off')  

    plt.tight_layout()  
    plt.show()

    return num_images

