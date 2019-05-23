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
 
ap = argparse.ArgumentParser(description = 'Use the trained model to make predictions')

ap.add_argument('input_image', nargs = '*', action = "store", type = str, default = "/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg")
ap.add_argument('checkpoint', nargs = '*', action = "store", type = str, default = "/home/workspace/ImageClassifier/checkpoint.pth")
ap.add_argument('--top_k', dest = "topk", action = "store", default = 3, type = int)
ap.add_argument('--cat_no_name' ,'-ctn', dest = "cat_name_dir", action = "store", default = 'cat_to_name.json')
ap.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
ap.add_argument('--arch' , dest = "pretrained_model", action = "store", default = 'vgg16')

pa = ap.parse_args()

print("Predict runs with model {}".format(pa.pretrained_model))

image = pa.input_image
path = pa.checkpoint
power = pa.gpu
top_k = pa.topk
cat = pa.cat_name_dir

pretr_model = pa.pretrained_model
model = getattr(models, pretr_model)(pretrained = True)
model_load = model_load(path, pa.pretrained_model)
    
with open (pa.cat_name_dir,'r') as json_file:
    cat_to_name = json.load(json_file)

probs, classes = predict(image, model_load, top_k)

print(probs)
print(classes)

names = []
for i in classes:
    names+= [cat_to_name[i]]
    
print(f"The image potentially contains a flower of category '{names[0]}' with the probability of {round(probs[0]*100,4)}%")

print("The prediction using the trained model has been made above, project ceompleted (hopefully)")