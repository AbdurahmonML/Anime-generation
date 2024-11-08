import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples=25000):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        all_images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(all_images) < num_samples:
            raise ValueError(f"Number of available images ({len(all_images)}) is less than the requested num_samples ({num_samples}).")
        self.images = np.random.choice(all_images, num_samples, replace=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
