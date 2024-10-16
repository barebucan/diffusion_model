import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()

        pos = torch.arange(0, time_steps, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.embedings = torch.zeros(time_steps, embed_dim, requires_grad=False).to(device)
        self.embedings[:, 0::2] = torch.sin(pos * div)
        self.embedings[:, 1::2] = torch.cos(pos * div)

    def forward(self, t):
        embeds = self.embedings[t].to(device)
        return embeds[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.gnorm1 = nn.GroupNorm(num_groups, C)
        self.gnorm2 = nn.GroupNorm(num_groups, C)
    
    def forward(self, x, embeddings):
        emb = embeddings[:, :x.shape[1], :, :]
        x = x + emb
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return x + r
    
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.linar_qvk = nn.Linear(C, C*3, bias=False)
        self.linar_out = nn.Linear(C, C, bias=False)
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, C)
        x = self.linar_qvk(x)
        x = x.view(batch_size, H*W,3, self.num_heads, C//self.num_heads)
        q = x[:, :, 0]
        k = x[:, :, 1]
        v = x[:, :, 2]
        q = q.permute(2, 0, 1, 3)  # (num_heads, batch_size, h*w, C//num_heads)
        k = k.permute(2, 0, 1, 3)  # (num_heads, batch_size, h*w, C//num_heads)
        v = v.permute(2, 0, 1, 3)  # (num_heads, batch_size, h*w, C//num_heads)
        x = F.scaled_dot_product_attention(q, k, v, is_causal =False, dropout_p= self.dropout_prob)
        x = x.permute(1, 2, 0, 3).contiguous().view(batch_size, H*W, C)
        x = self.linar_out(x)
        x = x.view(batch_size, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

class UnetLayer(nn.Module):
    def __init__(self, 
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout_prob: float,
                 C: int,
                 num_heads: int):
        super().__init__()
        self.res_block1 = ResBlock(C, num_groups, dropout_prob)
        self.res_block2 = ResBlock(C, num_groups, dropout_prob)
        if attention:
            self.attention = Attention(C, num_heads, dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding= 1)
        else:
            self.conv = nn.Conv2d(C, 2*C, kernel_size=3, stride = 2, padding=1)

    def forward(self, x, embeddings):
        x = self.res_block1(x, embeddings)
        if hasattr(self, 'attention'):
            x = self.attention(x)
        x = self.res_block2(x, embeddings)
        
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Upscales: List = [False, False, False, True, True, True],
            Attentions: List = [False, True, False, False, False, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            time_steps: int = 1000,
            input_channels: int = 1,
            output_channels: int = 1,
            label_embedding: bool = True):
        super().__init__()
        self.num_layers = len(Channels)
        self.inputconv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        for i in range(len(Channels)):
            layer = UnetLayer(Upscales[i], Attentions[i], num_groups, dropout_prob, Channels[i], num_heads)
            setattr(self, f'Layer{i+1}', layer)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.pos_embeddings = SinusoidalEmbeddings(time_steps, max(Channels))
        if label_embedding:
            self.label_embeddings = nn.Embedding(10, max(Channels))
        else:
            self.label_embeddings = None
    def forward(self, x, t, label = None):
        resiudal = []
        x = self.inputconv(x)
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            pos_embedding = self.pos_embeddings(t)
            embeddings = pos_embedding 
            if self.label_embeddings is not None:
                label_embedding = self.label_embeddings(label).unsqueeze(-1).unsqueeze(-1)
                embeddings += label_embedding  
            x, r = layer(x, embeddings)
            resiudal.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], resiudal[self.num_layers-i-1]), dim=1)

        return self.output_conv(self.relu(self.late_conv(x)))   

