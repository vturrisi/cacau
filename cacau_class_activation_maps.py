import os
import time
from collections import namedtuple
import cv2
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

class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        return (path, *original_tuple)

def prepare_data(dir_, data_transforms, batch_size=8, shuffle=True, num_workers=4):
    image_datasets = {x: ImageFolderWithPaths(os.path.join(dir_, x),
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
                       cmap=matplotlib.cm.coolwarm,
                       alpha=0.5,  # whole heatmap is translucent
                       annot=False,
                       zorder=2)
    hmax.imshow(img,
                zorder=1)  # put the map under the heatmap
    plt.axis('off')
    plt.savefig(name)
    plt.close()

def save_cam_cv(img_name, CAM, fname):
    CAM = np.uint8(255 * CAM)
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(fname, result)


class Flatten(nn.Module):
    def forward(self, imgs):
        size = imgs.size(1)
        return imgs.view(-1, size)

class PrintLayer(nn.Module):
    def forward(self, imgs):
        print(imgs.size())
        return imgs

class Resnet18(nn.Module):
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

class Resnet50(nn.Module):
    def __init__(self, base_resnet):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc_layer = nn.Linear(1024, n_classes, bias=False)

        self.activations.requires_grad = False
        self.gap.requires_grad = False

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output

def gen_activations(model, fname, img_name, img, label):
    activation_maps = model.activations(img).detach()
    b, c, h, w = activation_maps.size()
    activation_maps = activation_maps.view(c, h, w)
    weights = model.fc_layer.weight[label].detach().view(-1, 1, 1)
    activation_maps = activation_maps * weights
    # upscale before summing maps
    cam = torch.sum(activation_maps, 0)
    *_, i, j = cam.size()
    cam = cam.view(i, j)
    # cam = torch.abs(cam)  # absolute...
    min_v = torch.min(cam)
    max_v = torch.max(cam)
    range_v = max_v - min_v
    cam = (cam - min_v) / range_v
    cam = cam.numpy()
    cam = cv2.resize(cam, (224, 224))
    t = transforms.ToPILImage()
    img_cp = img.clone().view(3, 224, 224)
    # get back to original colour
    img_cp[0] = img_cp[0] * stds[0] + means[0]
    img_cp[1] = img_cp[1] * stds[1] + means[1]
    img_cp[2] = img_cp[2] * stds[2] + means[2]
    img_cp = t(img_cp)

    save_cam_cv(img_name, cam, fname)
    # save_cam(img_cp, cam, fname)

def gen_all_activations(device, model, model_name='model.bin'):
    since = time.time()

    # try to load model
    try:
        checkpoint = torch.load(model_name, map_location='cpu')
        model.load_state_dict(checkpoint['best_model_state_dict'])
    # file does not exist or pytorch error (model architecture changed)
    except Exception as e:
        raise Exception('Train a model first')

    model.eval()   # Set model to evaluate mode
    with torch.set_grad_enabled(False):
        for dataloader in dataloaders.values():
            # Iterate over data.
            for fname, img, label in dataloader:
                fname = fname[0]  # for some reason this is a tuple
                img = img.to(device)
                labels = label.to(device)
                base_fname = os.path.basename(fname)
                output = model(img)
                _, pred = torch.max(output, 1)
                pred = pred.item() + 1  # since its starts on 0
                gen_activations(model, 'CAM/predicted_{}_{}'.format(pred, base_fname), fname, img, label)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = create_transforms()

dwi = prepare_data('imgs', data_transforms, batch_size=1)
dataloaders = dwi.dataloaders
dataset_sizes = dwi.sizes
class_names = dwi.class_names
n_classes = dwi.n_classes

resnet = models.resnet18(pretrained=True).to(device)
model = Resnet18(resnet).to(device)

# resnet = models.resnet50(pretrained=True).to(device)
# model = Resnet50(resnet).to(device)


# for param in model.parameters():
#     param.requires_grad = False

# model.fc = nn.Linear(model.fc.in_features, n_classes)

criterion = nn.CrossEntropyLoss()

optimiser = optim.Adam(model.parameters(), lr=0.01)

model_ft = gen_all_activations(device, model, model_name='model_all_layers.bin')
