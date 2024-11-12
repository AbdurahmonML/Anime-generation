import gdown
import zipfile
import os
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np

def create_dataset(IMG_SIZE=64, BATCH_SIZE=8):
    # File ID from your Google Drive link
    file_id = '1zDm3Q4nWuk79V2VE-oWjmMPqTmzGCK4E'
    
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    output_file = 'archive2.zip'
    
    gdown.download(download_url, output_file, quiet=False)
    
    extracted_folder = 'extracted_images'
    
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)
    
    # Extract the ZIP file
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
    
    print(f"Files extracted to {extracted_folder}")
 
    # Set the image size
    IMG_SIZE = 64
    
    # Define data_transforms with additional transformations
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scale to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
    ])
    
    # StanfordCars dataset class
    class Anime(Dataset):
        def __init__(self, root_path, transform=None):
            self.images = [os.path.join(root_path, file) for file in os.listdir(root_path)]
            self.transform = transform
    
        def __len__(self):
            return len(self.images)
    
        def __getitem__(self, index):
            image_file = self.images[index]
            image = Image.open(image_file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
    
    # Initialize datasets with data_transforms
    train = Anime('extracted_images/images', transform=data_transforms)

    
    BATCH_SIZE = 8
    
    def load_transformed_dataset():
        data_transforms = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Scales data into [0,1] 
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ]
        data_transform = transforms.Compose(data_transforms)
    
        return torch.utils.data.ConcatDataset([train])
    def show_tensor_image(image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])
    
        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :] 
        plt.imshow(reverse_transforms(image))
    
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader
