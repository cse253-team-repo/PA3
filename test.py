from models import UNet,UNet_BN, FCN_backbone
from basic_fcn import FCN
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader,random_split
import numpy as np
import os
from tqdm import tqdm
import torch
from torchvision.models.segmentation import deeplabv3_resnet101,deeplabv3_resnet50
from utils import *
from utils import load_config
from torch.utils.tensorboard import SummaryWriter

# from tqdm import tqdm

import pdb

CUDA_DIX = [0,1,2,3,4]
class Test:
    def __init__(self,
                 config,
                 test_path = "./test.csv",
                 train_path = "./train.csv",
                 valid_path = "./val.csv",
                ):
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.num_classes = config["num_classes"]
        self.lr = config["lr"]
        self.opt_method = config["opt_method"]
        self.loss_method = config["loss_method"]
        self.save_best = config["save_best"]
        self.retrain = config["retrain"]
        GPU = config["GPU"]
        img_shape = tuple(config["img_shape"])
        model = config["model"]
        if GPU:
            self.gpus = CUDA_DIX

        else:
            self.gpus =[]
        self.record = SummaryWriter('runs/{}_{}'.format(model,time.time()))
        self.device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
        self.num_gpus = len(self.gpus)

        networks = {"UNet":UNet,
                    "base_fc":FCN,
                    "FCN":FCN_backbone,
                    "UNet_BN":UNet_BN,
                    "Deeplabv3":deeplabv3_resnet50
                    }
        self.model_name = model
        self.model = networks[self.model_name](num_classes = self.num_classes).to(self.device)


        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

        test_transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        self.valid_dst = CityScapesDataset(valid_path,transforms=test_transform)
        self.test_dst = CityScapesDataset(test_path,transforms=test_transform)
        print("Train set {}\n"
              "Validation set {}".format(
            len(self.valid_dst),
            len(self.test_dst)))

        self.valid_loader = DataLoader(self.valid_dst,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=1)
        self.test_loader = DataLoader(self.test_dst,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=1)

        self.load_weights(self.save_path)

    def test(self):
        valid_acc,valid_iou = self.check_accuracy(self.valid_loader)
        print("valid accuracy: {} \tvalid ious {}".format(valid_acc,valid_iou))

        # self.record.add_scalar("Validation miou", valid_miou.item(), epoch)

    def check_accuracy(self, dataloader):
        accs = []
        ious = []
        if os.path.exists('./result_images') == False:
            os.mkdir('./result_images')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                x, y_one_hot, y = data
                x = x.to(self.device)
                y_one_hot = y_one_hot.to(self.device)
                y = y.to(self.device)
                if self.model_name =="Deeplabv3":
                    out = self.model(x)["out"]
                else:
                    out = self.model(x)

                y_hat = torch.argmax(out,dim=1)
                visualize(y_hat,y,'./result_images/{}'.format(i))
                y_hat_onehot = to_one_hot(y_hat,self.num_classes)
                b_acc = pixel_acc(y_hat, y)
                b_ious = iou2(y_hat_onehot, y_one_hot)
                accs.append(b_acc)
                ious.append(b_ious)
                print('batch {}'.format(i))

        return np.mean(accs),np.mean(ious)

    def load_weights(self,path):
        print("Loading the parameters")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()



if __name__ == "__main__":
    config = load_config("base_fc_config.yaml")
    train = Test(config)
    train.test()


