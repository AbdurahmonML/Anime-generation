import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GeneratorModel(nn.Module):
    def __init__(self, input_dim=300):
        super(GeneratorModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 8 * 8 * 512),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=4, padding=1),
            nn.Sigmoid()
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        x = self.conv_layers(x)
        return x

# Define the Discriminator model
class DiscriminatorModel(nn.Module):
    def __init__(self, input_shape=(3, 64, 64)):
        super(DiscriminatorModel, self).__init__()

        # Define layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class DCGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=300):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        self.g_loss_metric = MeanMetric()
        self.d_loss_metric = MeanMetric()

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = real_images.size(0)

        random_noise = torch.randn(batch_size, self.latent_dim, device=real_images.device)


        self.discriminator.train()
        self.generator.eval()
        real_labels = torch.ones(batch_size, 1, device=real_images.device) * 0.95
        pred_real = self.discriminator(real_images)
        d_loss_real = self.loss_fn(pred_real, real_labels)

        fake_images = self.generator(random_noise).detach()
        fake_labels = torch.zeros(batch_size, 1, device=real_images.device)
        pred_fake = self.discriminator(fake_images)
        d_loss_fake = self.loss_fn(pred_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        self.generator.train()
        labels = torch.ones(batch_size, 1, device=real_images.device)
        fake_images = self.generator(random_noise)
        pred_fake = self.discriminator(fake_images)
        g_loss = self.loss_fn(pred_fake, labels)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        self.d_loss_metric.update(d_loss.item())
        self.g_loss_metric.update(g_loss.item())

        return {'d_loss': self.d_loss_metric.compute(), 'g_loss': self.g_loss_metric.compute()}
