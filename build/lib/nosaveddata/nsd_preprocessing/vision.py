import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# All transforms are applied over tensors, and NOT over PIL Images

def get_img_preprocessing_prob(batch_size, p, device):
    probs = torch.bernoulli(torch.ones(batch_size, device=device)*p)
    return probs[:,None,None,None]

def gray_scale(X, p=0.2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    gray_img = X.mean(1,keepdim=True).expand(-1,3,-1,-1)
    
    X = (1-probs)*X + probs*gray_img
    
    return X, probs.squeeze()

def gaussian_blur(X, p=0.2, sigma_min=0.1, sigma_max=2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    tfms = transforms.GaussianBlur(3, (sigma_min, sigma_max))
    
    blurred = tfms(X)
    X = (1-probs)*X + probs*blurred
    
    return X, probs.squeeze()

def solarization(X, p=0.2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    tfms = transforms.RandomSolarize(0,p=1) # This prob is applied over all the batch or no image at all
    
    solarized = tfms(X)
    X = (1-probs)*X + probs*solarized
    
    return X, probs.squeeze()


def preprocess_iwm(imgs, p=0.2):
    augments_applied=[]
    
    #imgs, augmented = gray_scale(imgs)
    #augments_applied.append(augmented)
    
    imgs, augmented = gaussian_blur(imgs, p)
    augments_applied.append(augmented)
    imgs, augmented = gaussian_blur(imgs, p)
    augments_applied.append(augmented)
    imgs, augmented = gaussian_blur(imgs, p)
    augments_applied.append(augmented)
    #imgs, augmented = solarization(imgs)
    #augments_applied.append(augmented)
    
    augments_applied = torch.stack(augments_applied,1)
    return imgs, augments_applied

def preprocess_iwm_no_solarize(imgs, p=0.2):
    augments_applied=[]
    
    imgs, augmented = gray_scale(imgs, p)
    augments_applied.append(augmented)
    
    imgs, augmented = gaussian_blur(imgs, p)
    augments_applied.append(augmented)
    
    augments_applied = torch.stack(augments_applied,1)
    return imgs, augments_applied


def gray_scale_stacked(X, p=0.2, stacks=4):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    stacked_probs = probs.repeat_interleave(stacks,0)
    X = X.view(-1,X.shape[1]//stacks,*X.shape[-2:])
    
    gray_img = X.mean(1,keepdim=True).expand(-1,3,-1,-1)
    
    X = (1-stacked_probs)*X + stacked_probs*gray_img
    
    return X.view(X.shape[0]//stacks, -1, *X.shape[-2:]), probs.squeeze()

def gaussian_blur(X, p=0.2, stacks=4, sigma_min=0.1, sigma_max=2):
    # Input: Tensor T e (B,C,T,D)
    
    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    tfms = transforms.GaussianBlur(3, (sigma_min, sigma_max))
    
    blurred = tfms(X)
    X = (1-probs)*X + probs*blurred
    
    return X, probs.squeeze()

def solarization_stacked(X, p=0.2, stacks=4):
    # Input: Tensor T e (B,C,T,D)

    probs = get_img_preprocessing_prob(X.shape[0], p, X.device)
    stacked_probs = probs.repeat_interleave(stacks,0)
    
    X = X.view(-1,X.shape[1]//stacks,*X.shape[-2:])
    
    tfms = transforms.RandomSolarize(0,p=1) # This prob is applied over all the batch or no image at all
    
    solarized = tfms(X)
    X = (1-stacked_probs)*X + stacked_probs*solarized
    
    return X.view(X.shape[0]//stacks, -1, *X.shape[-2:]), probs.squeeze()


def preprocess_iwm_stacked(imgs, p=0.2, stacks=4):
    # Applies the same preprocessing for all images in the sequence, but separated by each beach
    augments_applied=[]
    
    imgs, augmented = gray_scale_stacked(imgs, p, stacks)
    augments_applied.append(augmented)
    
    imgs, augmented = gaussian_blur(imgs, p, stacks)
    augments_applied.append(augmented)
    
    imgs, augmented = solarization_stacked(imgs, p, stacks)
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