import torch
import torchvision
import numpy as np
from torchvision import transforms
import os
import pandas as pd
from torch.utils import data
root= '/home/kowshik/DomainBed/DATA/'
train_transform =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
celeba_train = torchvision.datasets.CelebA(root= root, split='train',transform=train_transform, target_type='attr')
celeba_val = torchvision.datasets.CelebA(root= root, split='valid',transform=test_transform, target_type='attr')
celeba_test = torchvision.datasets.CelebA(root= root, split='test',transform=test_transform, target_type='attr')
attribute_df = pd.read_csv(os.path.join(root,'celeba','list_attr_celeba.txt'),delim_whitespace=True, skiprows=1)
attributes= attribute_df.columns.tolist()
attribute_dict= dict(zip(attributes,np.arange(len(attributes))))
# a= next(iter(data.DataLoader(celeba_test,batch_size=3)))
# a