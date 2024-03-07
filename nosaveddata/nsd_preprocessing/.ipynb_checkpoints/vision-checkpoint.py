import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# All transforms are applied over tensors, and NOT over PIL Images

def get_img_preprocessing_prob(batch_size, p):
    probs = torch.bernoulli(torch.ones(batch_size)*p)
    return probs[:,None,None,None]

def gray_scale(X, p=0.2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p)
    gray_img = X.mean(1,keepdim=True).expand(-1,3,-1,-1)
    
    X = (1-probs)*X + probs*gray_img
    
    return X, probs.squeeze()

def gaussian_blur(X, p=0.2, sigma_min=0.1, sigma_max=2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p)
    tfms = transforms.GaussianBlur(3, (sigma_min, sigma_max))
    
    blurred = tfms(X)
    X = (1-probs)*X + probs*blurred
    
    return X, probs.squeeze()

def solarization(X, p=0.2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p)
    tfms = transforms.RandomSolarize(0,p=1) # This prob is applied over all the batch or no image at all
    
    solarized = tfms(X)
    X = (1-probs)*X + probs*solarized
    
    return X, probs.squeeze()


def preprocess_iwm(imgs):
    augments_applied=[]
    
    imgs, augmented = gray_scale(imgs)
    augments_applied.append(augmented)
    
    imgs, augmented = gaussian_blur(imgs)
    augments_applied.append(augmented)
    
    imgs, augmented = solarization(imgs)
    augments_applied.append(augmented)
    
    augments_applied = torch.stack(augments_applied,1)
    return imgs, augments_applied

def preprocess_iwm_no_solarize(imgs):
    augments_applied=[]
    
    imgs, augmented = gray_scale(imgs)
    augments_applied.append(augmented)
    
    imgs, augmented = gaussian_blur(imgs)
    augments_applied.append(augmented)
    
    augments_applied = torch.stack(augments_applied,1)
    return imgs, augments_applied



def plot_img(x):
    assert x.shape[-1]==3, 'Channels must be the last dimension'
    assert len(x.shape)==3, 'Use plot_imgs(x) instead'
    
    plt.axis('off')
    plt.imshow(x)
    
def plot_imgs(x):
    assert x.shape[-1]==3, 'Channels must be the last dimension'
    assert len(x.shape)>3, 'Use plot_img(x) instead'
    
    num_images = x.shape[0]

    # Set up the subplot grid
    num_rows = 1
    num_cols = num_images

    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Loop through each image and plot it
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x[i])
        plt.axis('off')  # Turn off axis labels

    plt.show()