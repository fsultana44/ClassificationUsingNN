from __future__ import print_function, division
import os
from torch import nn
import torch,torchvision
import pandas as pd
import pandas
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

root_dirA= '/home/fatema/Downloads/homework/datasetA/'
root_dirB= '/home/fatema/Downloads/homework/datasetB/'
classfile =pd.read_csv('/home/fatema/Downloads/homework/baseline.csv')

# ID = classfile.iloc[:, 1]
# side = classfile.iloc[:, 2]
# grade=classfile.iloc[:, 6]


class AOIDataset(Dataset):



    def __init__(self, csv_file, root_dir,transform=None):

        self.classfile = pd.read_csv(csv_file, header=None)
        self.ID = self.classfile.iloc[:, 1]
        self.SIDE = self.classfile.iloc[:, 2]
        self.V00XRKL = self.classfile.iloc[:, 6]
        self.root_dir = root_dir
        self.transform = transform
        # self.transform2 = transform2

    def __len__(self):
        return len(self.classfile)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.ID[idx])

        # Read the numpy files
        patches, p_id = np.load(img_name)
        print(patches, p_id .shape)
        img_class = self.V00XRKL[idx]
        img_ID = self.ID[idx]
        img_side =self.SIDE[idx]
        sample ={'subject id':img_ID,'side':img_side,'class':img_class}
        # Return image and the label
        if self.transform:
            sample = self.transform(sample)

        return sample


trans = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


dset_train = AOIDataset('/home/fatema/Downloads/homework/baseline.csv',root_dirA,transform=trans)
dset_test = AOIDataset('/home/fatema/Downloads/homework/baseline.csv',root_dirB,transform=trans)

Train_loader = torch.utils.data.DataLoader(dset_train,
                                           batch_size=10,
                                           num_workers=0,
                                           shuffle=False)

Test_loader = torch.utils.data.DataLoader(dset_train,
                                          batch_size=10,
                                          num_workers=0,
                                          shuffle=False)

