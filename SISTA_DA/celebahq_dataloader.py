import glob
import os
import os.path
import random
import sys
from os import path
from pprint import pprint
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.datasets import CelebA

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

class CelebaHQDataset(data.Dataset):
    def __init__(self,
                domain='photo',
                attribute='Smiling', 
                data_dir='DATA/CelebA-HQ-split',
                train=True,
                transform=None,
                train_size=None,
                return_index=False,
                ):
        super(CelebaHQDataset, self).__init__()

        mode = 'train' if train else 'test'
        mode_d='img_align_celeba'

        self.img_files = glob.glob(f'{os.path.join(os.path.join(data_dir,mode), domain)}/*.jpg')
        
        total_imgs = len(self.img_files)

        if train_size is not None:
            total_imgs = train_size
        
        self.img_files = self.img_files[:total_imgs]

        self.attribute = attribute
        self.att_table = pd.read_csv(f'{data_dir}/CelebAMask-HQ-attribute-anno.csv')#,delim_whitespace=True)
        #self.att_table.index.name="Name"
        #self.att_table.reset_index(inplace=True)
        self.att_table=self.att_table[["Name",self.attribute]]
        
        self.transform = transform
        self.return_index = return_index
        

    def __getitem__(self, index):
        image = Image.open(self.img_files[index]).convert('RGB')
        
        file_name = os.path.split(self.img_files[index])[1]

        try:
            row_interest = self.att_table.loc[self.att_table['Name'] == file_name].values.tolist()[0]
        except:
            print(self.img_files[index])
            assert False

        class_label = 0 if row_interest[1]<0 else 1 
        

        if self.transform is not None:
            image = self.transform(image)
        if self.return_index:
            return image, class_label, index
        return image, class_label

    def __len__(self):
        return len(self.img_files)

class JOJOGanDataset(data.Dataset):
    def __init__(self,
                domain='photo', 
                data_dir='/home/kowshik/generative_sfda/DATA/multi_domain_interpol_gen_10K',
                interp_weights=[0],
                transform=None,
                train_size=None,
                return_index=False
                ):
        super(JOJOGanDataset, self).__init__()
        self.img_files = []
        
        
        for interp_weight in interp_weights:
            self.img_path=os.path.join(data_dir,domain+'_'+str(interp_weight))
            img_files = [os.path.join(self.img_path,f) for f in sorted(os.listdir(self.img_path)) if f.endswith('.png')]
            self.img_files.append(img_files)
        self.img_files= np.concatenate(self.img_files)
        total_imgs = len(self.img_files)

        if train_size is not None:
            total_imgs = train_size
            np.random.shuffle(self.img_files)
        
        self.img_files = self.img_files[:total_imgs]
        self.transform = transform
        self.return_index = return_index
        

    def __getitem__(self, index):
        
        image = Image.open(self.img_files[index]).convert('RGB')
        
        class_label = -1000 

        if self.transform is not None:
            image = self.transform(image)
        if self.return_index:
            return image, class_label, index
        return image, class_label

    def __len__(self):
        return len(self.img_files)
class RefImgs(JOJOGanDataset): 
    def __init__(self,
                domain='photo', 
                data_dir='/home/kowshik/generative_sfda/DATA/ref_imgs',
                
                transform=None,
                train_size=None,
                return_index=True
                ):
        self.img_files=[]
        self.img_path= os.path.join(data_dir,domain)
        img_files = [os.path.join(self.img_path,f) for f in sorted(os.listdir(self.img_path)) if f.endswith('.jpg')]
        self.img_files.append(img_files)
        self.img_files =np.concatenate(self.img_files)
        self.transform = transform
        self.return_index = return_index
        self.train_size= train_size
        super(JOJOGanDataset, self).__init__()
    




class JOJOGANPruneRewind(JOJOGanDataset):
    def __init__(self, domain = 'sketch',
                data_dir = '/home/kowshik/generative_sfda/DATA/prune_rewind',
                interp_weights= [0],
                transform=None,
                train_size=None,
                return_index=False,
                prune=True,
                rewind=True):
        self.img_files=[]
        prune_imgs=[]
        rewind_imgs=[]
        
        for interp in [0,1]:
            prune_dir= os.path.join(data_dir,'prune','prune_0_'+domain+'_'+str(interp))
            prune_imgs.append(np.array([os.path.join(prune_dir,f) for f in sorted(os.listdir(prune_dir)) if f.endswith('.png')]))
            
        
        # for interp in [0,1]:
        #     rewind_dir= os.path.join(data_dir,'rewind','rewind_'+domain+'_'+str(interp))
        #     rewind_imgs.append(np.array([os.path.join(rewind_dir,f) for f in sorted(os.listdir(rewind_dir)) if f.endswith('.png')]))
        #rewind_dir= os.path.join(data_dir,'prune_followed_rewind','rewind_'+domain)
        rewind_dir= os.path.join(data_dir,'prune_followed_rewind',domain)
        rewind_imgs.append(np.array([os.path.join(rewind_dir,f) for f in sorted(os.listdir(rewind_dir)) if f.endswith('.png')]))

        image_ids= lambda x: [np.random.randint(1000,size=x) for _ in range(len(prune_imgs))]

        if (prune and rewind):
            # number of images =250 each
            
            prune_imgs_ids = image_ids(250)
            rewind_imgs_ids= [np.random.randint(1000,size=500)]
            prune_imgs_new = np.concatenate(\
                            [prune_imgs[i][prune_imgs_ids[i]] for i in range(len(prune_imgs))]\
                            )
            rewind_imgs_new = np.concatenate(\
                            [rewind_imgs[i][rewind_imgs_ids[i]] for i in range(len(rewind_imgs_ids))]\
                            )
            self.img_files = np.concatenate((prune_imgs_new,rewind_imgs_new))

        elif (prune and not rewind):
            #number of images =500 each
            prune_imgs_ids = image_ids(500)
            prune_imgs_new = np.concatenate(\
                            [prune_imgs[i][prune_imgs_ids[i]] for i in range(len(prune_imgs))]\
                            )

            self.img_files = prune_imgs_new
        elif (rewind and not prune):
            rewind_imgs_ids= [np.arange(1000)]
            rewind_imgs_new = np.concatenate(\
                            [rewind_imgs[i][rewind_imgs_ids[i]] for i in range(len(rewind_imgs_ids))]\
                            )
            self.img_files = rewind_imgs_new
        else: 
            raise ValueError(" Atleast one among prune and rewind should be specified")

        # now also get the JOJOGAN generated images
        data_dir='/home/kowshik/generative_sfda/DATA/multi_domain_interpol_gen_2'
        img_path=os.path.join(data_dir,domain+'_'+str(0)) 
        img_files = np.array(\
                    [os.path.join(img_path,f) for f in sorted(os.listdir(img_path)) if f.endswith('.png')]\
                    )
        self.img_files = np.concatenate((img_files,self.img_files))
       
        self.transform = transform
        self.return_index = return_index

        super(JOJOGanDataset, self).__init__()


class JOJOGanDatasetMultipleImages(data.Dataset):
    """
    For the same index - load and retur multiple images from different folder
    i.e, target_with_weight 0 and along with it other weights
    
    """
    def __init__(self,
                domain='photo', 
                data_dir='/home/kowshik/generative_sfda/DATA/multi_domain_interpol_gen_2',
                interp_weights=[0],
                transform=None,
                train_size=None,
                return_index=False,
                ):
        super(JOJOGanDatasetMultipleImages, self).__init__()
        self.img_files ={}
        self.interp_weights =interp_weights
        for interp_weight in interp_weights:
            self.img_path=os.path.join(data_dir,domain+'_'+str(interp_weight))
            img_files = [os.path.join(self.img_path,f) for f in sorted(os.listdir(self.img_path)) if f.endswith('.png')]
            self.img_files[str(interp_weight)] = img_files
        
        
        total_imgs = len(self.img_files[str(0)])


        if train_size is not None:
            total_imgs = train_size
        
        self.img_files = self.img_files[:total_imgs]


        
        self.transform = transform
        self.return_index = return_index
        

    def __getitem__(self, index):
        filenames = [v[index] for k,v in self.img_files.items()]
        image = [Image.open(f).convert('RGB') for f in filenames]
        
        class_label = -1000 
        

        if self.transform is not None:
            image = [self.transform(f) for f in image]
        interp_weights_list = ['interp_weight_'+str(f) for f in self.interp_weights]
        image = dict(zip(interp_weights_list,image))
        if self.return_index:
            return image, class_label, index
        return image, class_label

    def __len__(self):
        return len(self.img_files['0'])


if __name__ == '__main__':
    # dset = JOJOGanDatasetMultipleImages('color_sketch',interp_weights=[0,2,4],transform= train_transform,return_index=True)
    # dloader = data.DataLoader(dset,batch_size=5)
    # for data in dloader:
    #     data
    dset = JOJOGANPruneRewind('watercolor',\
        transform= train_transform,return_index=True,prune=False,rewind=True)
    print(len(dset))
    dloader = data.DataLoader(dset,batch_size=5,shuffle=True)
    for i,data in enumerate(dloader):

       print(data)
       break
    # dset = RefImgs('color_sketch',transform= train_transform,return_index=True)
    # print(len(dset))
#celeba = torchvision.datasets.CelebA(root='.', split ='train',  download=True)
