from PIL import Image
import torch
import torchvision
import numpy as np

def show_spectrogram_and_estimate(spectrogram, estimate):
    img = torchvision.transforms.ToPILImage()(torch.hstack((spectrogram, estimate)))
    img.show()
    return img