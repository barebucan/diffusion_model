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
from torchvision.models import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BACTH_SIZE = 8
LR = 2e-5
NUM_EPOCHS = 75
EMA_DECAY = 0.9999
NUM_TIME_STEPS = 1000

class Classifier:
    def __init__(self, weights_path):
        # Initialize ResNet18 model and modify the final layer for 10 classes
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # STL10 has 10 classes
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(weights_path))
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(device)
    
    def get_class_guidance(self, images, labels):

        images.requires_grad = True
        
        # Forward pass through the classifier to get class logits
        logits = self.model(images)
        
        # Compute the cross-entropy loss between predicted logits and true labels
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        # Backward pass to compute the gradients (class guidance)
        self.model.zero_grad()  # Zero out previous gradients
        loss.backward()
        
        # Obtain the gradients of the images with respect to the loss
        class_guidance = images.grad
        
        return class_guidance


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

class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps = 1000):
        super().__init__()
        self.beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False).to(device)
        aplha_t = 1 - self.beta_t
        self.alpha_t = torch.cumprod(aplha_t, dim=0).requires_grad_(False).to(device)

    def call(self, t):
        return self.alpha_t[t], self.beta_t[t]

TRANS = transforms.Compose([
    transforms.RandomCrop(96, padding=4),   # Random crop with padding to simulate cropping
    transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
    transforms.ToTensor(),                  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def train_classifier(num_epochs, LR):

    dataset = torchvision.datasets.STL10(root='stl10_data', split='train', download=True, transform=TRANS)

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        correct = 0
        total = 0

        for i, (x, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(x)
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # No need to calculate gradients for validation
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)

                # Forward pass
                output = model(x)
                loss = criterion(output, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    #save
    torch.save(model.state_dict(), 'classifier_weights.pth')

def train(checkpoint_path = None):
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    train_data =  torchvision.datasets.STL10(root='stl10_data', split= 'train', transform=transforms.ToTensor())
    # train_data = torch.utils.data.Subset(train_data, list(range(0, len(train_data), 10)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BACTH_SIZE, shuffle=True)
    
    model = UNET(
            time_steps= NUM_TIME_STEPS,
            input_channels = 3,
            output_channels = 3,
            label_embedding= False)
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
            t = torch.randint(0, NUM_TIME_STEPS, (x.shape[0],), requires_grad=False)
            a = scheduler.alpha_t[t].to(device).view(-1, 1, 1, 1)
            e = torch.randn_like(x, requires_grad=False).to(device)
            optimizer.zero_grad()
            x = (torch.sqrt(a) * x + torch.sqrt(1-a) * e)

            output = model(x, t)
            loss = criterion(output, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            sum_loss += loss.item()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {epoch+1} | Loss {sum_loss / (60000/BACTH_SIZE):.5f}')

        if epoch % 5 == 0:


            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, 'Unet96_checkpoint.pt')
            
            validation(ema, epoch)
def display(images):

    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = x.permute(1, 2, 0)  # Rearrange to (height, width, channels)
        x = x.cpu().numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()

def save_img(images, epoch):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        print(torch.max(x).item())
        x = x.permute(1, 2, 0)  # Rearrange to (height, width, channels)
        x = x.cpu().numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(f'img{epoch}.png')

def validation(ema, epoch):

    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []
    with torch.no_grad():
        model = ema.module.eval() 
        z = torch.randn(1, 3, 96, 96).to(device)
        for t in reversed(range(1, NUM_TIME_STEPS)):
            t = [t]
            temp = (scheduler.beta_t[t]/( (torch.sqrt(1-scheduler.alpha_t[t]))*(torch.sqrt(1-scheduler.beta_t[t])) ))
            z = (1/(torch.sqrt(1-scheduler.beta_t[t])))*z - (temp*model(z.cuda(),t))
            if t[0] in times:
                images.append(z)
            e = torch.randn_like(z).to(device)
            z = z + (e*torch.sqrt(scheduler.beta_t[t]))
        temp = scheduler.beta_t[0]/( (torch.sqrt(1-scheduler.alpha_t[0]))*(torch.sqrt(1-scheduler.beta_t[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta_t[0])))*z - (temp*model(z.cuda(),[0]))

        images.append(x)
        save_img(images, epoch)

def inference(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    model = UNET(
            time_steps= NUM_TIME_STEPS,
            input_channels = 3,
            output_channels = 3,
            label_embedding= False).to(device)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV2(model, decay=EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)
    times = [0, 3,10,20,40,60,80,110,140, 199]
    images = []
    train_data =  torchvision.datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            
            z = torch.randn(1, 3, 96, 96).to(device)
            for t in reversed(range(1, NUM_TIME_STEPS)):
                t = [t]
                temp = (scheduler.beta_t[t]/( (torch.sqrt(1-scheduler.alpha_t[t]))*(torch.sqrt(1-scheduler.beta_t[t])) ))
                z = (1/(torch.sqrt(1-scheduler.beta_t[t])))*z - (temp*model(z.cuda(),t))
                if t[0] in times:
                    images.append(z)
                e = torch.randn_like(z).to(device)
                z = z + (e*torch.sqrt(scheduler.beta_t[t]))
            temp = scheduler.beta_t[0]/( (torch.sqrt(1-scheduler.alpha_t[0]))*(torch.sqrt(1-scheduler.beta_t[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta_t[0])))*z - (temp*model(z.cuda(),[0]))

            images.append(x)
            display(images)
            images = []

train()
# inference("Unet96_checkpoint.pt")