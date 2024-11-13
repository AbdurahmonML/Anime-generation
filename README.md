# **Anime Face Generation**

## **Exploring and Comparing VAE, Autoencoder, GAN, and Diffusion Models**

In this project, I set out to generate anime faces using **four different models**: Variational Autoencoder (VAE), Autoencoder, Generative Adversarial Network (GAN), and Diffusion Model. Each model approaches the task of generation differently, offering unique characteristics in quality, diversity, and style. Here, I will test and compare them to determine which one performs best in generating high-quality anime faces.

Each model includes:
- A **sample result** of generated images.
- A **demo notebook** (`demo.ipynb`) that you can run to train and test the models yourself.

---

## Project Overview

### GAN:
The Generative Adversarial Network (GAN) uses adversarial training, where a generator and discriminator compete to produce more realistic faces. This results in sharper images, though GANs can be more challenging to train effectively.

**Results of generating:**
![image](https://github.com/user-attachments/assets/886e4eaa-9775-4c09-8213-139bed3846e4)
![image](https://github.com/user-attachments/assets/ff26523e-3384-4aab-9250-8045ac18970a)

- **Advantages**: Relatively fast to train and generate new images; High quality images can be generated;
- **Disadvantages**: Difficult to configure due to sensitivity to hyperparameters; Prone to instability.
- **Demo**: [GAN demo.ipynb](./GAN/GAN_demo.ipynb)

- **Trained notebook**: If you are not able to open GAN/GAN_training.ipynb, then open this file to see training, as I have code of all files in this [notebook](https://drive.google.com/file/d/19YW5mzsTPKO1i2SuNskB3mX-f6NmMpm8/view?usp=sharing)

### Diffusion Model:
After experimenting with a non-attention-based diffusion model and getting suboptimal results, I decided to modify the architecture by incorporating attention mechanisms to better capture long-range dependencies and improve the quality of the generated outputs. This change aims to enhance the model's ability to focus on relevant features during the diffusion process.


**Results of generating:**
![image](https://github.com/user-attachments/assets/d54a6a6f-3c29-4092-b3d7-efe87fafaa69)

- **Advantages**: High quality, Stability
- **Disadvantages**: Very slow to train and generate new images.
- **Demo**: [Diffusion Model demo.ipynb](./Diffusion/diff_demo.ipynb)
- **Trained notebook**: [notebook](https://drive.google.com/file/d/1oRt-ekHGpjgfTH_wz8ANhCf3e-L4MF4L/view?usp=sharing)
- **Weights after 35-th epoch**: [Weights](https://drive.google.com/file/d/1fQoEAaiVJeBYbmZorFwp6sJOSeIHx6ze/view?usp=drive_link)
  
### VAE:
The Variational Autoencoder (VAE) encodes images into a latent space and allows controlled generation by sampling. It produces diverse anime faces with smooth transitions across the latent space, although with less sharpness than other models.
- **Advantages**: Stable and reliable training; Efficient latent space representation (we can at least generate images from random noise)
- **Disadvantages**: Lower image quality
  
**Results of generating:**

When I sample purely random vectors:
![image](https://github.com/user-attachments/assets/f91527a8-b9b8-42ab-befc-c6477b014030)






When I sample random vectors using the training images' mean and standard deviation:
![image](https://github.com/user-attachments/assets/451cf30f-453e-43cb-a8a0-77dc8aa85a5a)




- **Demo**: [vae_demo.ipynb](./VAE/vae_demo.ipynb)

### Autoencoder:
The Autoencoder captures the general structure of anime faces through reconstruction, focusing more on regenerating input images than creating new variations. It’s a simpler model and provides good baseline results.
- **Advantages**: Stable and easy to train
- **Disadvantages**: Lower image quality; Limited generative capability (we get Very bad result if we generate image from random noise)
  
**Results of generating:**

When I sample purely random vectors:
![image](https://github.com/user-attachments/assets/44c26af1-63ba-413b-ba34-75acbb91db93)


When I sample random vectors using the training images' mean and standard deviation: 
![image](https://github.com/user-attachments/assets/e998dd49-e59f-4f6a-a8f1-81615682d82b)

- **Demo**: [autoencoder_demo.ipynb](./Autoencoder/autoencoder_demo.ipynb)

### Dataset: 
https://www.kaggle.com/datasets/splcher/animefacedataset/data


### How to Run?

You can easily reproduce the results by running the provided notebooks for each model. Each notebook contains clear instructions to guide you through the process. Simply follow the steps in the notebooks to experiment with the models and generate anime faces.

### Conclusion

After experimenting with different models for anime face generation, the results indicate the following rankings based on quality and diversity:

1. **GAN (Generative Adversarial Network)**: The GAN model produces the sharpest and most realistic images, making it the best performer for generating high-quality anime faces. Its adversarial training allows the generator to refine the images, producing impressive results.

2. **Diffusion Model**: The Diffusion Model, especially after incorporating attention mechanisms, comes in second. It generates diverse and visually appealing faces, though it doesn't quite match the sharpness of GAN-generated images. However, its ability to capture long-range dependencies gives it a unique edge.

3. **VAE (Variational Autoencoder)**: The VAE generates a good variety of faces and exhibits smooth transitions across the latent space, making it a strong contender for controlled generation. However, the output images tend to lack the sharpness seen in GANs or Diffusion Models.

4. **Autoencoder**: The Autoencoder, while producing reasonable results, is more focused on reconstructing input images rather than generating new variations. As a result, it ranks last, offering good baseline performance but lacking the diversity and realism of the other models. Also, using Autoencoder we cannot sample fully random vector and generate image, because it will give bad result, but VAE works better here.

I trained the diffusion model for 35 epochs, and while additional training might slightly improve the results, it is unlikely to surpass the performance of the GAN model in terms of quality.
