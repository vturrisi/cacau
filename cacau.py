import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def create_transforms():
    size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5,
            #                        saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }
    return data_transforms


def prepare_data(dir_, data_transforms, batch_size=8, shuffle=True, num_workers=4):
    from collections import namedtuple

    DataWithInfo = namedtuple('DataWithInfo', ['dataloaders', 'sizes',
                                           'class_names', 'n_classes'])

    image_datasets = {x: datasets.ImageFolder(os.path.join(dir_, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    n_classes = len(class_names)

    dwi = DataWithInfo(dataloaders=dataloaders, sizes=dataset_sizes,
                       class_names=class_names, n_classes=n_classes)
    return dwi


def load_model(model_name):
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state'])
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    best_model_wts = checkpoint['best_model_wts']
    best_acc = checkpoint['best_acc']
    return epoch, best_model_wts, best_acc


def train_model(model, criterion, optimizer, scheduler,
                num_epochs=25, model_name='cur_state_cacau'):
    global device

    since = time.time()

    if model_name in os.listdir():
        epoch, best_model_wts, best_acc = load_model(model_name)
    else:
        epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    while epoch < num_epochs:
        # save current model to resume training later
        cur_state = {'state': model.state_dict(),
                     'best_model_wts': best_model_wts,
                     'best_acc': best_acc,
                     'epoch': epoch,
                     'scheduler': scheduler.state_dict(),
                     'optimizer': optimizer.state_dict()}
        torch.save(cur_state, model_name)

        with open(outfile, 'a') as f:
            f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            f.write('-' * 10 + '\n')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            with open(outfile, 'a') as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                    phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch += 1
        with open(outfile, 'a') as f:
            f.write('\n')
        print()

    time_elapsed = time.time() - since
    with open(outfile, 'a') as f:
        f.write('Training complete in {:.0f}m {:.0f}s'.format(
              time_elapsed // 60, time_elapsed % 60))
        f.write('Best val Acc: {:4f}'.format(best_acc))

    print('Training complete in {:.0f}m {:.0f}s\n'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}\n'.format(best_acc))


    cur_state = {'state': model.state_dict(),
                 'best_model_wts': best_model_wts,
                 'best_acc': best_acc,
                 'epoch': epoch,
                 'scheduler': scheduler.state_dict(),
                 'optimizer': optimizer.state_dict()}
    torch.save(cur_state, model_name)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = create_transforms()

dwi = prepare_data('imgs', data_transforms)
dataloaders = dwi.dataloaders
dataset_sizes = dwi.sizes
class_names = dwi.class_names
n_classes = dwi.n_classes

outfile = 'log.txt'

model_ft = models.resnet18(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

model_ft.fc = nn.Linear(512, n_classes) # resnet18
# model_ft.fc = nn.Linear(25088, n_classes) # resnet34
# model_ft.fc = nn.Linear(100352, n_classes) # resnet50


model = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=10e-4)

scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

model_ft = train_model(model, criterion, optimizer_ft, scheduler,
                       num_epochs=220, model_name='resnet18')
