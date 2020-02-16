import os

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from model.ASPP import Deeplab
from model.basic_fcn import FCN
from model.models import UNet, UNet_BN, FCN_backbone
from utils.dataloader import *
from utils.utils import *

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
        if "CUDA_DIX" in config:
            self.CUDA_DIX = config["CUDA_DIX"]
        else:
            self.CUDA_DIX = [0]
        if "visualize" in config:
            self.visualize = config["visualize"]
        else:
            self.visualize = True
        GPU = config["GPU"]
        img_shape = tuple(config["img_shape"])
        model = config["model"]
        if GPU:
            self.gpus = self.CUDA_DIX

        else:
            self.gpus =[]

        self.device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
        self.num_gpus = len(self.gpus)

        networks = {"UNet":UNet,
                    "base_fc":FCN,
                    "FCN":FCN_backbone,
                    "UNet_BN":UNet_BN,
                    "Deeplabv3":deeplabv3_resnet50,
                    "Deeplab": Deeplab
                    }
        self.model_name = model

        if model=="Deeplab":
            self.model = networks[self.model_name](num_classes = self.num_classes, use_torch_model=config["use_torch_model"],
                                                retrain_backbone=config["retrain_backbone"],
                                                 backbone=config["backbone"]).to(self.device)
        else:
            self.save_path = "my_model_{}.pt".format(model)
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

        # self.new_val,_ = random_split(self.valid_dst,[10,len(self.valid_dst)-10])
        self.valid_loader = DataLoader(self.valid_dst,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=1)
        self.test_loader = DataLoader(self.test_dst,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=1)
        print(self.save_path)
        self.load_weights(self.save_path)

    def test(self):
        valid_acc,valid_iou = self.check_accuracy(self.valid_loader)
        print("valid accuracy: {} \t valid ious {}".format(valid_acc,valid_iou))

        # self.record.add_scalar("Validation miou", valid_miou.item(), epoch)

    def check_accuracy(self, dataloader):
        accs = []
        if os.path.exists('./result_images') == False:
            os.mkdir('./result_images')
        self.model.eval()
        ioucomputer = IOU(self.num_classes)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                x, y_one_hot, y = data
                x = x.to(self.device)
                y_one_hot = y_one_hot.to(self.device)
                y = y.to(self.device)
                if self.model_name =="Deeplabv3":
                    out = self.model(x)["out"]
                else:
                    out = self.model(x)

                y_hat = torch.argmax(out, dim=1)
                if self.visualize:
                    visualize(y_hat,y,'./result_images/{}'.format(i))
                y_hat_onehot = to_one_hot(y_hat, self.num_classes).to(self.device)

                b_acc = pixel_acc(y_hat, y)
                ioucomputer.UpdateIou(y_hat_onehot, y_one_hot)
                print(b_acc)
                accs.append(b_acc)
                print('batch {}'.format(i))
        accs = np.array(accs)
        ious = np.array(ioucomputer.CalculateIou())
        print(ious)
        return np.mean(accs),np.mean(ious[~np.isnan(ious)])

    def load_weights(self,path):
        print("Loading the parameters")
        self.model.load_state_dict(torch.load(path))
        self.model.eval()



if __name__ == "__main__":
    config = load_config("config/aspp.yaml")
    print(config)
    train = Test(config)
    train.test()


