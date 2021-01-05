import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline

import torch
from torch.utils.data import Dataset


# https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
class PairMeansDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        game = self.data.iloc[index]  # .values.astype(np.uint8).reshape((1, 28, 28))
        # try:
        #     game.drop(['num',"PlayerA","PlayerB","NumOfGames","WinRatio"],inplace=True)
        # except:
        label = self.data.iloc[index]["label"].astype(np.longlong)
        game.drop(['label', 'PlayerA', 'PlayerB', 'PlayerC'], inplace=True, errors='ignore')

        # label = self.data.iloc[index, 5].astype(np.longlong)
        game = torch.tensor(game.values.astype(np.float32))
        if self.transform is not None:
            game = self.transform(game)

        return game, label

    def get_player_a_name(self, index):
        name = self.data.iloc[index]["PlayerA"]
        return name

    def get_atter(self, index, atter):
        name = self.data.iloc[index][atter]
        return name

    def get_player_b_name(self, index):
        name = self.data.iloc[index]["PlayerB"]
        return name

    def get_player_c_name(self, index):
        name = self.data.iloc[index]["PlayerC"]
        return name
