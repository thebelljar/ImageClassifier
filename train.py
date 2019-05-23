import matplotlib.pyplot as plt
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
import argparse
import json

import function
import tools

from function import load_data, process_image
from tools import build_classifier, validation, train_model, test_model, save_model, model_load, predict

ap = argparse.ArgumentParser(description = 'train.py')

ap.add_argument('data_dir', nargs = '*', action = "store", default = "./flowers/")
ap.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
ap.add_argument('--save_dir', '-sd', dest = "save_dir", action = "store", default = "./checkpoint.pth")
ap.add_argument('--learning_rate', '-lr', dest = "learning_rate", action = "store", default = 0.0001)
ap.add_argument('--dropout', '-do', dest = "dropout", action = "store", default = 0.2)
ap.add_argument('--epochs', '-e' , dest = "epochs", action = "store", type = int, default = 1)
ap.add_argument('--hidden_units', '-hu', type=int, dest="hidden_units", action="store", default=10)
ap.add_argument('--arch' , dest = "pretrained_model", action = "store", default = 'vgg16')


pa = ap.parse_args()

where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
save_dir = pa.save_dir
dropout = pa.dropout
power = pa.gpu
epochs = pa.epochs
architecture = pa.pretrained_model
hiddenl = pa.hidden_units


trainloader, validloader, testloader, train_data, valid_data, test_data = load_data(where)


pretr_model = pa.pretrained_model
model = getattr(models, pretr_model)(pretrained = True)

build_classifier(model)
build_classifier(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=0.0001)

model, optimizer = train_model(model, epochs, trainloader, validloader, criterion, optimizer, power, lr, hiddenl, dropout)

test_model(model, testloader)
save_model(model, train_data, optimizer, save_dir, epochs, lr, architecture, hiddenl, dropout)

print("The Model is trained") 