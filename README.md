# **Anime Face Generation**

## **Exploring and Comparing VAE, Autoencoder, GAN, and Diffusion Models**

In this project, I set out to generate anime faces using **four different models**: Variational Autoencoder (VAE), Autoencoder, Generative Adversarial Network (GAN), and Diffusion Model. Each model approaches the task of generation differently, offering unique characteristics in quality, diversity, and style. Here, I will test and compare them to determine which one performs best in generating high-quality anime faces.

Each model includes:
- A **sample result** of generated images.
- A **demo notebook** (`demo.ipynb`) that you can run to train and test the models yourself.

---

## Project Overview

### VAE:
The Variational Autoencoder (VAE) encodes images into a latent space and allows controlled generation by sampling. It produces diverse anime faces with smooth transitions across the latent space, although with less sharpness than other models.

**Results:**

When I sample purely random vectors:
![image](https://github.com/user-attachments/assets/f91527a8-b9b8-42ab-befc-c6477b014030)






When I sample random vectors using the training images' mean and standard deviation:
![image](https://github.com/user-attachments/assets/451cf30f-453e-43cb-a8a0-77dc8aa85a5a)




- **Demo**: [vae_demo.ipynb](./VAE/vae_demo.ipynb)

### Autoencoder:
The Autoencoder captures the general structure of anime faces through reconstruction, focusing more on regenerating input images than creating new variations. It’s a simpler model and provides good baseline results.

**Results:**
When I sample purely random vectors:
![image](https://github.com/user-attachments/assets/44c26af1-63ba-413b-ba34-75acbb91db93)


When I sample random vectors using the training images' mean and standard deviation: 
![image](https://github.com/user-attachments/assets/e998dd49-e59f-4f6a-a8f1-81615682d82b)

- **Demo**: [autoencoder_demo.ipynb](./Autoencoder/autoencoder_demo.ipynb)

### GAN:
The Generative Adversarial Network (GAN) uses adversarial training, where a generator and discriminator compete to produce more realistic faces. This results in sharper images, though GANs can be more challenging to train effectively.

**Results:**
![image](https://github.com/user-attachments/assets/886e4eaa-9775-4c09-8213-139bed3846e4)
![image](https://github.com/user-attachments/assets/5ccdf30e-4a77-4625-86b7-8fe426207927)


- **Demo**: [GAN demo.ipynb](./gan/demo.ipynb)

- **Trained notebook**: If you are not able to open GAN/GAN_training.ipynb, then open this file to see training, as I have code of all files in this [notebook](https://drive.google.com/file/d/19YW5mzsTPKO1i2SuNskB3mX-f6NmMpm8/view?usp=sharing) :

### Diffusion Model:
The Diffusion Model gradually removes noise from random images to create high-quality, detailed anime faces. This method generates the most intricate and realistic samples but is resource-intensive.

**Results:**
![Diffusion Model Results](./results/diffusion_sample.png)

- **Demo**: [Diffusion Model demo.ipynb](./diffusion/demo.ipynb)

---

## Getting Started

### Prerequisites
- **Python 3.7+**
- Libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm` (and any additional requirements listed in `requirements.txt`)

Install dependencies:
```bash
pip install -r requirements.txt
