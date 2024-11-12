import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

# Function to train model for one epoch
def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch, total_epochs):
    model.train()
    
    # Initialize the total loss
    total_loss = 0.0
    num_batches = 0

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{total_epochs}")
        
        for x0s in loader:
            x0s = x0s.cuda()
            ts = torch.randint(low=1, high=1000, size=(x0s.shape[0],), device='cuda')
            xts, gt_noise = sd(x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate the loss and increment batch count
            total_loss += loss.detach().item()
            num_batches += 1

        # Calculate the mean loss over all batches
        mean_loss = total_loss / num_batches
        print('mean_loss', mean_loss, 'epoch=', epoch)

    return mean_loss
