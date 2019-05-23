import numpy as np
import seaborn as sns

import torch
import PIL
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle= True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)

    # Get original dimensions
    orig_width, orig_height = image.size

    ratio = orig_width/orig_height

    # Find shorter size and create settings to crop shortest side to 256
    if ratio >= 1: n_dim=[256, int(256*ratio)]
    else: n_dim=[int(256*ratio), 256]
    
    new_width = n_dim[0]
    new_height = n_dim[1]
    crop_dim = (round((new_width - 224)/2), round((new_height - 224)/2),round((new_width - 224)/2)+224, round((new_height - 224)/2)+ 224)
     
    
    image = image.resize(n_dim)
    image = image.crop(crop_dim)

    np_image = np.array(image)
    np_image = np_image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image
