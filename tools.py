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

from function import process_image

def build_classifier(model):
    for param in model.parameters():
        param.requires_grad=False

    # Define a new feedforward network for use as a classifier using the features as input, using Relu activation function and Dropout to minimise overfitting

    classifier = nn. Sequential (OrderedDict([
                                ('fc1', nn.Linear(25088, 5000)),
                                ('relu1', nn.ReLU()),
                                ('do', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(5000, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    model.classifier = classifier
    
    return model

def validation(model, validloader, criterion, valid_loss):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
    
        images, labels = images.cuda(), labels.cuda()
        
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train_model(model, epochs, trainloader, validloader, criterion, optimizer , power, lr, hiddenl, dropout):
    #epochs = 5
    print_every = 40
    steps = 0

    model = model.cuda()
    # Implementing back-propogation to obtain the features
    print( "------Training Commencing-------")
    for e in range(epochs):
        running_loss = 0
        valid_loss = 0
        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():

                    valid_loss, accuracy = validation(model, validloader, criterion, valid_loss)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()  
    return model, optimizer

def test_model(model, testloader):
    correct = 0
    total=0
    model.cuda()

    print( "------Testing Commencing-------")
    with torch.no_grad():
        for data in testloader:
            images, labels=data

            images, labels = images.cuda(), labels.cuda()

            outputs=model(images)
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on the test dataset: %d %%' % (100*(correct/total)))
    
def save_model(model, train_data, optimizer, save_dir, epochs,lr, architecture, hiddenl, dropout):
    
    print( "----------Saving the checkpoint, with model:")
    checkpoint = {
                'classifier': model.classifier,
                'optimizer':optimizer.state_dict(),
                'model_dict': model.state_dict(),
                'class_to_idx': train_data.class_to_idx}

    return torch.save(checkpoint, save_dir)

def model_load(path, pretrained_model):
    
    checkpoint = torch.load(path)
    if pretrained_model == 'vgg16':
        model = models.vgg16(pretrained= True)
    elif pretrained_model == 'vgg11':
        model = models.vgg11(pretrained= True)
    else:
        print( "Incorrect architecture choice, please choose between vgg11 or vgg16")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
   
    return model


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    image = process_image(image_path)
    image = torch.from_numpy(image).float().to('cpu')

    image_tensor = image.type(torch.FloatTensor)
    image_tensor_plus = image_tensor.unsqueeze_(0)

    model.eval()
    model.to('cpu')
    
    with torch.no_grad():
        output = model.forward(image_tensor_plus)


    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [idx_to_class[index]]
    
    return probs_top_list, classes_top_list   

  
