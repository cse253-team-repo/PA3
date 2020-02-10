from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
import torch.functional as F
from PIL import Image
from collections import namedtuple

n_class    = 19
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                      # during evaluations or not

    'color'       , # The color of this label
    ] )

labels_classes = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) )
]

class CenterCrop(object):
    def __init__(self,arg):
        self.transform = transforms.CenterCrop(arg)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),self.transform(label)

class Resize(object):
    def __init__(self,arg):
        self.transform = transforms.Resize(arg,Image.NEAREST)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),self.transform(label)

class Normalize(object):
    def __init__(self,mean,std):
        self.transform = transforms.Normalize(mean, std)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),label

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.id_to_trainId = {}
        for label in labels_classes:
            self.id_to_trainId[label.id]=label.trainId if label.trainId != 255 else -1
        self.id_to_trainId_map_func = np.vectorize(self.id_to_trainId.get)
    def __call__(self, sample):
        img, label = sample
        label = np.array(label)
        label = self.id_to_trainId_map_func(label)
        return self.transform(img), \
               torch.from_numpy(label.copy()).long()

class RandomResizedCrop(object):
    def __init__(self,size,
                 scale=(0.8, 1.2),
                 ratio=(3. / 4., 4. / 3.)):
        self.scale = scale
        self.radio = ratio
        self.size = size

    def __call__(self, sample):
        img, label = sample
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, self.scale,self.radio)

        img = transforms.functional.resized_crop(img, i, j, h, w)
        label = transforms.functional.resized_crop(label, i, j, h, w, self.size,Image.NEAREST)
        return img,label

class RandomCrop(object):
    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self, sample):
        img, label = sample

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.output_size)

        img = transforms.functional.crop(img, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)
        return img,label

class CityScapesDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms=None):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        # Add any transformations here
        self.transform = transforms

        if self.transform==None:
            self.transform = "resize"

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img = Image.open(img_name)

        label_name = self.data.iloc[idx, 1]
        label = Image.open(label_name)

        if self.transform != None:
            img,label = self.transform((img,label))

        # create one-hot encoding
        h, w = label.shape[0], label.shape[1]
        target = torch.zeros(self.n_class, h, w).long()

        for c in range(self.n_class):
            target[c][label == c] = 1

        return img, target, label

if __name__ == "__main__":

    # hard coding
    h,w = 1024, 2048

    crop_method = 'random'

    if crop_method == 'center':
        transform = transforms.Compose([
            CenterCrop((h//2,w//2)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
            ])
    elif crop_method == 'resize':
        transform = transforms.Compose([
            Resize((h//2,w//2)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
    elif crop_method == 'random':
        transform = transforms.Compose([
            RandomCrop((h//2,w//2)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
            ])


    trainset = CityScapesDataset("train.csv", transforms=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, num_workers=4, batch_size=64, shuffle=False)
    print("train loader: ", len(train_loader))
    print("cuda: ", torch.cuda.is_available())
    for i, (img, target, label) in enumerate(trainset):
        img = img.cuda()
        target = target.cuda()
        label = label.cuda()
        # print(torch.cuda.memory_allocated())
        print("img shape: {}, target shape: {}, label shape: {}".format(img.shape, target.shape, label.shape) )
        break
