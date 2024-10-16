import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from unet import UNET
import os
from tqdm import tqdm
from timm.utils import ModelEmaV2

NUM_EPOCHS = 75
BACTH_SIZE = 128
LR = 2e-5
EMA_DECAY = 0.9999
NUM_TIME_STEPS = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps = 1000):
        super().__init__()
        self.beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False).to(device)
        aplha_t = 1 - self.beta_t
        self.alpha_t = torch.cumprod(aplha_t, dim=0).requires_grad_(False).to(device)

    def call(self, t):
        return self.alpha_t[t], self.beta_t[t]
    
class DDPM_beta_t_cosine_scheduler(nn.Module):
    def __init__(self, num_steps = 1000, s = 0.008):
        super().__init__()
        self.T = num_steps
        self.s = s
        self.beta_t, self.alphas, self.alpha_t = self.compute_betas_and_alphas()
    

    def compute_betas_and_alphas(self):
        
        t = torch.arange(0, self.T + 1).requires_grad_(False).to(device)
        
        # Compute the cumulative noise schedule (alpha_bar) using the cosine function
        alpha_bar = torch.cos(((t / self.T + self.s) / (1 + self.s)) * torch.pi / 2) ** 2
        
        # Compute betas: beta_t = 1 - (alpha_bar[t] / alpha_bar[t-1])
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0, 0.999)
        
        # Compute alphas: alpha_t = 1 - beta_t
        alphas = 1 - betas
        
        # Compute the cumulative product of alphas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        return betas, alphas, alpha_cumprod

def train(checkpoint_path = None):
    scheduler = DDPM_beta_t_linear_scheduler(NUM_TIME_STEPS)

    train_data =  torchvision.datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BACTH_SIZE, shuffle=True)
    
    model = UNET()
    model = model.to(device)
    
    ema = ModelEmaV2(model, decay = EMA_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean')

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(NUM_EPOCHS):
        sum_loss = 0
        for i, (x, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            x = x.to(device)
            labels = labels.to(device)
            t = torch.randint(0, NUM_TIME_STEPS, (x.shape[0],), requires_grad=False)
            a = scheduler.alpha_t[t].to(device).view(-1, 1, 1, 1)
            x = F.pad(x, (2,2,2,2))
            e = torch.randn_like(x, requires_grad=False).to(device)
            optimizer.zero_grad()
            x = (torch.sqrt(a) * x + torch.sqrt(1-a) * e)

            output = model(x, t, labels)
            loss = criterion(output, e)
            loss.backward()
            sum_loss += loss.item()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {epoch+1} | Loss {sum_loss / (60000/BACTH_SIZE):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pt')
def display(images):

    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = x.permute(1, 2, 0)  # Rearrange to (height, width, channels)
        x = x.cpu().numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()

def inference(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV2(model, decay=EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []
    train_data =  torchvision.datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            
            z = train_data[i][0].unsqueeze(0).to(device)
            z = F.pad(z, (2,2,2,2))
            label_z = train_data[i][1]
            # z = torch.randn(1, 1, 32, 32).to(device)
            label = torch.randint(0, 10, (1,)).to(device)
            while label_z == label:
                label = torch.randint(0, 10, (1,)).to(device)
            print("Generating image of ", label.item(), " from image of ", label_z)
            for t in reversed(range(1, NUM_TIME_STEPS)):
                t = [t]
                temp = (scheduler.beta_t[t]/( (torch.sqrt(1-scheduler.alpha_t[t]))*(torch.sqrt(1-scheduler.beta_t[t])) ))
                z = (1/(torch.sqrt(1-scheduler.beta_t[t])))*z - (temp*model(z.cuda(),t, label))
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 1, 32, 32).to(device)
                z = z + (e*torch.sqrt(scheduler.beta_t[t]))
            temp = scheduler.beta_t[0]/( (torch.sqrt(1-scheduler.alpha_t[0]))*(torch.sqrt(1-scheduler.beta_t[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta_t[0])))*z - (temp*model(z.cuda(),[0], label))

            images.append(x)
            display(images)
            images = []

def main():
    inference('checkpoint.pt')


if __name__ == '__main__':
    main()
