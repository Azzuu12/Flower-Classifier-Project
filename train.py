# PROGRAMMER:Ezzeddine Almansoob
# DATE CREATED:21/7/2023
# PURPOSE: here we can get all the argemnts from user command line interface

#import related functions and libraries
from train_Models import get_model
from train_Arg import get_ar
import torch
from torch import nn, optim
from torchvision import datasets, transforms


def main():
    #This section is for transforamtion and processing the dataset
    in_arg = get_ar()
    train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406]
                                                                , [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406]
                                                                , [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(in_arg.data_dir + '/train',
                                         transform=train_transforms)
    valid_dataset = datasets.ImageFolder(in_arg.data_dir + '/valid',
                                         transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                              shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)

    #Calling the get_model() function to build the desired model
    model = get_model(in_arg.arch, in_arg.hidden_units)

    #Prepreare the parametrs needed for training process
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    #This is an option tif you want to use GUI
    device = torch.device('cuda' if torch.cuda.is_available() and in_arg.gpu else 'cpu')
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    model.to(device)
    model.train()
    #Training process
    for e in range(epochs):
        steps += 1
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if steps == epochs:
            #Validation process
            accuracy = 0
            valid_loss = 0
            with torch.no_grad():
                model.eval()
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)
                    loss = criterion(log_ps, labels)
                    valid_loss += loss.item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc = torch.mean(equals.type(torch.FloatTensor))
                    accuracy += acc.item()
            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
                  "Accuracy: {:.3f}".format(accuracy / len(validloader)))

            running_loss = 0
    model.class_to_idx = train_dataset.class_to_idx
    if in_arg.save_dir:
        torch.save({'model': in_arg.arch,
                    'hidden_units': in_arg.hidden_units,
                    'state_dict': model.classifier.state_dict(),
                    'class_to_idx': model.class_to_idx,
                    'optimizer_state_dict': optimizer.state_dict()}
                   , '{}/checkpoint.pth'.format(in_arg.save_dir))


if __name__ == "__main__":
    main()
