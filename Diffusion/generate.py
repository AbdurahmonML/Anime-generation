import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate(model, sd, num_images=8, timesteps=1000, nrow=16, format='png', device="cpu"):
    generate_video = False

    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.{format}"

    save_path = os.path.join(filename)
    reverse_diffusion(
        model,
        sd,
        num_images=num_images,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=timesteps,
        img_shape=(3, 64, 64),
        nrow=nrow,
        device=device
    )
    print(save_path)
generate(model, sd, format='png', device=device)
