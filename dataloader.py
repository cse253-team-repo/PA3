from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd

n_class    = 34
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

class CityScapesDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms=None):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        # Add any transformations here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]

        img = np.asarray(Image.open(img_name).convert('RGB'))
        label_name = self.data.iloc[idx, 1]
        label      = np.asarray(Image.open(label_name))

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.shape
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        return img, target, label