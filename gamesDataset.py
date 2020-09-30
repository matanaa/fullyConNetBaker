import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

#https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
class GamesDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path,low_memory=False)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        game = self.data.iloc[index]#.values.astype(np.uint8).reshape((1, 28, 28))
        game.drop(['Result',"pa_Result","pa_ResultType","pb_Result","pa_Id","pb_Id","Game_id"],inplace=True)
        # game.drop(['Result',"pa_Result","pa_ResultType","pb_Result","pa_Id","pb_Id","Game_id","Win_Rate"],inplace=True)
        # label = self.data.iloc[index]['Win_Rate'].astype(np.longlong)
        label = self.data.iloc[index]['Result'].astype(np.longlong)
        try:
            game = torch.tensor(game.values.astype(np.float32))
        except:
            pass
        if self.transform is not None:
            game = self.transform(game)

        return game, label



    def get_player_a_name(self,index):
        name = self.get_atter(index,"pa_Id")
        return name

    def get_player_b_name(self,index):
        name = self.get_atter(index,"pb_Id")
        return name

    def get_atter(self,index,atter):
        name = self.data.iloc[index][atter]
        return name