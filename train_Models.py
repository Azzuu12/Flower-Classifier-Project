# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:20/7/2023
# REVISED DATE:
# PURPOSE: create a model based on a pretrained model but with change its classifier if requested

#import related libraries and functions
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms, models

#import pretrained models
densenet121 = models.densenet121(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
models_ls = {'vgg19': vgg19,'densenet121': densenet121, 'alexnet': alexnet}

def get_model(model_name, hidden_units):
    if hidden_units == 0:
        #create new model based on a pretrained model
        model = models_ls[model_name]
    # create new model based on a pretrained model but with new classifier
    else:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            for para in model.parameters():
                para.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(1024, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=0.4)),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
            model.classifier =classifier

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            for para in model.parameters():
                para.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(9216, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=0.4)),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
            model.classifier =classifier

        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
            for para in model.parameters():
                para.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(25088, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=0.4)),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
            model.classifier =classifier
    return model