import torch
import matplotlib.pyplot as plt

def get_num_parameter(model):
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def display_image(image):
    plt.figure(figsize=(10.5, 10.5), facecolor='white')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
