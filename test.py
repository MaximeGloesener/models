# Imports 
import torch.nn.functional as F
from utils.benchmark import *
import torch
import torch_pruning as tp
import sys
import os
from models.vgg import VGG
import torchprofile
import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from torchprofile import profile_macs
assert torch.cuda.is_available()

# Fixer le seed pour la reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()

# Datas
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        #Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
    "test": Compose([
        ToTensor(),
        #Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
}



# récupérer les jeux de données d'entrainement et de test
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )

from models.resnet import resnet56


# model initial 
print('model initial')
checkpoint = torch.load('models/vgg.cifar.pretrained.pth', map_location="cpu")
model = VGG()
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda')
model.to(device)
example_input = torch.rand((32,3,224,224))
acc, loss = evaluate(model, dataloader['test'])
print(f'accuracy = {acc:.2f} % / loss = {loss:.2f}')
benchmark(model, example_input, plot = False)
print('')



# pruned model
print('pruned model')
model = torch.load("results/speed_up_20.0_cifar_vgg.pth", map_location="cpu")
model.eval()
device = torch.device('cuda')
model.to(device)
example_input = torch.rand((32,3,224,224)) 
acc, loss = evaluate(model, dataloader['test'])
print(f'accuracy = {acc:.2f} % / loss = {loss:.2f}')
benchmark(model, example_input, plot = False)

with torch.inference_mode():
    benchmark(model, example_input, plot = False)

"""for f in os.listdir('results'):
    if f.endswith('.pth'):
        print(f)
        model = torch.load('results/' + f)
        model.eval()
        model.to(device)
        acc, loss = evaluate(model, dataloader['test'])
        print(f'accuracy = {acc:.2f} % / loss = {loss:.2f}')
        benchmark(model, example_input)
        print('')"""
