import os
import time
from collections import namedtuple

import matplotlib.cm
import matplotlib.colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

sns.set()


DataWithInfo = namedtuple('DataWithInfo',
                          ['dataloaders', 'sizes', 'class_names', 'n_classes'])
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

def create_transforms():
    global means, stds
    size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
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


def save_cam(img, cam, name):
    hmax = sns.heatmap(cam,
                       cmap=matplotlib.cm.bwr,
                       alpha=0.5,  # whole heatmap is translucent
                       annot=False,
                       zorder=2)

    hmax.imshow(img,
                aspect=hmax.get_aspect(),
                extent=hmax.get_xlim() + hmax.get_ylim(),
                zorder=1)  # put the map under the heatmap
    plt.axis('off')
    plt.savefig(name)
    plt.close()


class Flatten(nn.Module):
    def forward(self, imgs):
        size = imgs.size(1)
        return imgs.view(-1, size)

class PrintLayer(nn.Module):
    def forward(self, imgs):
        print(imgs.size())
        return imgs

class ResnetWithCAM(nn.Module):
    def __init__(self, base_resnet):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc_layer = nn.Linear(256, n_classes, bias=False)

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output

    def get_activations(self, img, label):
        activation_maps = self.activations(img).detach()
        weights = self.fc_layer.weight[label].detach()
        activation_maps = activation_maps * weights.view(1, -1, 1, 1)
        # upscale before summing maps
        activation_maps = F.interpolate(activation_maps, 224)
        cam = torch.sum(activation_maps, 1)
        *_, i, j = cam.size()
        cam = cam.view(1, 1, i, j)
        # upscale after summing maps
        # cam = F.interpolate(cam, 224)
        min_v = torch.min(cam)
        max_v = torch.max(cam)
        range_v = max_v - min_v
        cam = (cam - min_v) / range_v
        cam = cam.view(224, 224).numpy()

        t = transforms.ToPILImage()
        img_cp = img.clone().view(3, 224, 224)
        # get back to original colour
        img_cp[0] = img_cp[0] * stds[0] + means[0]
        img_cp[1] = img_cp[1] * stds[1] + means[1]
        img_cp[2] = img_cp[2] * stds[2] + means[2]
        img_cp = t(img_cp)
        save_cam(img_cp, cam, 'test.jpg')
        exit()
        return

def train_model(device, model, criterion, optimiser, epochs, model_name='model.bin'):
    since = time.time()

    # try to load model
    try:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['best_model_state_dict'])
    # file does not exist or pytorch error (model architecture changed)
    except:
        raise Exception('Train a model first')

    for phase in ['val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                model(inputs)
                act = model.get_activations(inputs[0].view(1, 3, 224, 224), labels[0])
                return

    time_elapsed = time.time() - since
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


# for param in model.parameters():
#     param.requires_grad = False

# model.fc = nn.Linear(model.fc.in_features, n_classes)

criterion = nn.CrossEntropyLoss()

optimiser = optim.Adam(model.parameters(), lr=0.01)

model_ft = train_model(device, model, criterion, optimiser,
                       epochs=200, model_name='model.bin')
