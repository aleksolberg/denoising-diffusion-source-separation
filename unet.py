from torch import nn
import torch
import math
import ssutils
import torchvision

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UNet(nn.Module):
    def __init__(self, channels = 1, num_layers = 6, kernel_size = (5,5), stride = 2, padding=True, second_layer_channels=16, dropout=0.5, spectrogram_condition = True):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = tuple(ti//2 for ti in self.kernel_size) if padding is True else padding
        self.second_layer_channels=second_layer_channels
        if type(dropout) == (float or int):
            self.dropout = [dropout] * self.num_layers
        elif type(dropout) == list:
            '''if len(dropout) != self.num_layers - 1:
                raise IndexError('Dropout list is not of correct length.')
            else:'''
            self.dropout = dropout
        else:
            raise TypeError('Dropout must be either of type \'float\' or \'list\'.')
        self.spectrogram_condition = spectrogram_condition
        self.in_channels = channels * (2 if spectrogram_condition else 1)

        # Encoder
        self.convolutional_layers = nn.ModuleList()
        layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.second_layer_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(self.second_layer_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.convolutional_layers.append(layer)
        for i in range(self.num_layers - 1):
            layer = nn.Sequential(
                nn.Conv2d(self.second_layer_channels*2**i, self.second_layer_channels*2**(i+1), kernel_size = self.kernel_size, stride=self.stride, padding=self.padding),
                nn.BatchNorm2d(self.second_layer_channels*2**(i+1)),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.convolutional_layers.append(layer)
        
        # Decoder
        layer = nn.ConvTranspose2d(self.second_layer_channels*2**(self.num_layers-1), self.second_layer_channels*2**(self.num_layers-2), 
                                   kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
        self.deconvolutional_layers = nn.ModuleList()
        self.deconvolutional_layers.append(layer)
        for i in range(self.num_layers - 1, 1, -1):
            layer = nn.ConvTranspose2d(self.second_layer_channels*2**i, self.second_layer_channels*2**(i-2), 
                                       kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
            self.deconvolutional_layers.append(layer)
        layer = nn.ConvTranspose2d(self.second_layer_channels*2, self.in_channels, 
                                   kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
        self.deconvolutional_layers.append(layer)

        if self.spectrogram_condition:
            self.last_conv = nn.Conv2d(self.in_channels, 1, kernel_size=1)


        self.deconvolutional_BAD_layers = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            layer = nn.Sequential(
                nn.BatchNorm2d(self.second_layer_channels*2**(i-1)),
                nn.ReLU(),
                nn.Dropout2d(self.dropout[self.num_layers - (i+1)])
            )
            self.deconvolutional_BAD_layers.append(layer)


        # Time embedding
        self.time_emb_dim = self.second_layer_channels*4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.time_emb_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.Sequential(
                nn.Linear(self.time_emb_dim, self.second_layer_channels*2**i),
                nn.ReLU()
            )
            self.time_emb_layers.append(layer)
        for i in range(self.num_layers - 2, 0, -1):
            layer = nn.Sequential(
                nn.Linear(self.time_emb_dim, self.second_layer_channels*2**(i-1)),
                nn.ReLU()
            )
            self.time_emb_layers.append(layer)
        
    def forward(self, data, times, y_cond = None):
        in_shape = data.shape
        t = self.time_mlp(times)
        if self.spectrogram_condition:
            data = torch.cat((y_cond, data), dim=1)
        conv = [data]

        for i in range(self.num_layers):
            data = self.convolutional_layers[i](data)
            time_emb = self.time_emb_layers[i](t)
            data += time_emb[:,:,None,None]
            conv.append(data)

        data = self.deconvolutional_layers[0](data, output_size = conv[self.num_layers-1].size())
        for i in range(1, self.num_layers-1):
            data = self.deconvolutional_layers[i](torch.cat([data, conv[self.num_layers - i]], 1), output_size = conv[self.num_layers - (i+1)].size())
            data = self.deconvolutional_BAD_layers[i](data)
            time_emb = self.time_emb_layers[self.num_layers + i - 1](t)
            data += time_emb[:,:,None,None]

        data = self.deconvolutional_layers[-1](torch.cat([data, conv[1]], 1), output_size = in_shape)
        if self.spectrogram_condition:
            data = self.last_conv(data)

        return data
    
'''spectrogram = torch.ones(8,1,256,256)/2
noise = torch.rand(8,1,256,256)
times = torch.ones(8)

model = UNet()
data = model(noise, times, spectrogram)
print(data)
'''