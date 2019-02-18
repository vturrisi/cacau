import os
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


DataWithInfo = namedtuple('DataWithInfo',
                          ['dataloaders', 'sizes', 'class_names', 'n_classes'])

def create_transforms():
    size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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


class Flatten(nn.Module):
    def forward(self, imgs):
        size = imgs.size(1)
        return imgs.view(-1, size)

class ResnetWithCAM(nn.Module):
    def __init__(self, base_resnet):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc_layer = nn.Linear(256, n_classes, bias=False)

        self.activations.requires_grad = False
        self.gap.requires_grad = False

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output

    def get_activations(self, img, label):
        activation_maps = self.activations(img).detach()
        weights = self.fc_layer.weight[label].detach()
        activation_maps = activation_maps * weights.view(1, -1, 1, 1)

        exit()


def train_model(device, model, criterion, optimiser, epochs, model_name='model.bin'):
    since = time.time()

    # try to load model
    try:
        checkpoint = torch.load(model_name)
        initial_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        best_acc = checkpoint['best_acc']
        best_model_state_dict = checkpoint['best_model_state_dict']
    # file does not exist or pytorch error (model architecture changed)
    except:
        initial_epoch = 0
        best_acc = 0

    for epoch in range(initial_epoch, epochs):
        # save current model to resume training later
        if epoch != 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'best_acc': best_acc,
                        'best_model_state_dict': best_model_state_dict,
                        }, model_name)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_loss = 0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimiser.zero_grad()
                        loss.backward()
                        optimiser.step()

                # statistics
                epoch_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.item() / len(dataloaders[phase].dataset)

            print('(Epoch #{} - {}) Loss: {:.4f} Acc: {:.4f}'.format(
                  epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = model.state_dict()

    time_elapsed = time.time() - since

    print('Training completed in {:.0f}m {:.0f}s\n'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_state_dict)
    return model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = create_transforms()

dwi = prepare_data('imgs', data_transforms, batch_size=64)
dataloaders = dwi.dataloaders
dataset_sizes = dwi.sizes
class_names = dwi.class_names
n_classes = dwi.n_classes

resnet = models.resnet18(pretrained=True).to(device)
model = ResnetWithCAM(resnet).to(device)

# model.fc = nn.Linear(model.fc.in_features, n_classes)

criterion = nn.CrossEntropyLoss()

optimiser = optim.Adam(model.fc_layer.parameters(), lr=0.01)

model_ft = train_model(device, model, criterion, optimiser,
                       epochs=200, model_name='model.bin')
