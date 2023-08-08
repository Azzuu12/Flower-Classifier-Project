# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:25/7/2023
# PURPOSE: here we can get all the argemnts from user command line interface

#import related functions and libraries
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict

def process_image(image):
    #Process the image
    img_pil = Image.open(image)
    re_gen = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_ten = re_gen(img_pil)
    return img_ten

#Function to build the model with option to load the model that you trained before using train.py
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hidden_units=checkpoint['hidden_units']
    if checkpoint['model']=='vgg19':
        model=models.vgg19(pretrained=True)
        for para in model.parameters():
            para.requires_grad=False
        classifier=nn.Sequential(OrderedDict([
                         ('fc1',nn.Linear(25088,hidden_units)),
                         ('relu1',nn.ReLU()),
                         ('dropout1',nn.Dropout(p=0.4)),
                         ('fc2',nn.Linear(hidden_units,102)),
                         ('output',nn.LogSoftmax(dim=1))]))
        model.classifier=classifier
        model.classifier.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx=checkpoint['class_to_idx']

    elif checkpoint['model']=='densenet121':
        model = models.densenet121(pretrained=True)
        for para in model.parameters():
            para.requires_grad=False
        classifier=nn.Sequential(OrderedDict([
                         ('fc1',nn.Linear(1024,hidden_units)),
                         ('relu1',nn.ReLU()),
                         ('dropout1',nn.Dropout(p=0.4)),
                         ('fc2',nn.Linear(hidden_units,102)),
                         ('output',nn.LogSoftmax(dim=1))]))
        model.classifier=classifier
        model.classifier.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx=checkpoint['class_to_idx']

    elif checkpoint['model'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        for para in model.parameters():
            para.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.4)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier
        model.classifier.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    return model