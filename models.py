import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

pretrained_models = {
            'resnext50':torchvision.models.resnext50_32x4d,
            'resnext101':torchvision.models.resnext101_32x8d,
			'resnet18': torchvision.models.resnet18,
			'resnet34': torchvision.models.resnet34,
			'resnet50': torchvision.models.resnet50,
			'resnet101': torchvision.models.resnet101,
			'vgg11_bn': torchvision.models.vgg11_bn,
			'vgg16_bn': torchvision.models.vgg16_bn,
			'vgg19_bn': torchvision.models.vgg19_bn
		}
encoder_out_chnnel={
			'resnet18': 512,
			'resnet34': 1024,
			'resnet50': 2048,
            'resnext50': 2048,
			'resnet101': 2048,
			'resnext101': 2048,
			'vgg11_bn': 512,
			'vgg16_bn': 512,
			'vgg19_bn': 512
}
class Loss:
    def __init__(self, method="cross-entropy"):
        """
            Implement various loss function inside this class inclusing naive cross-entropy
            and a loss weighting scheme.
        """
        if method == "cross-entropy":
            self.loss = self.cross_entropy
    def cross_entropy(self, y, target):
        pass





class UNet_BN(nn.Module):
    def __init__(self, num_classes):
        super(UNet_BN, self).__init__()
        # scaling factor of the network size
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True)
                                    )
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(128, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )
        self.layer4 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(256, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True)
                                    )
        self.layer5 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(512, 1024, 3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, 3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True)
                                    )
        self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(1024, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True)
                                    )
        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(512, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.layer7 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )
        self.deconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(256, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.layer8 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )
        self.deconv4 =  nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(128,64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.layer9 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, num_classes, 1)
                                    )

    def forward(self, x):
        en_x1 = self.layer1(x)
        en_x2 = self.layer2(en_x1)
        en_x3 = self.layer3(en_x2)
        en_x4 = self.layer4(en_x3)
        en_x5 = self.layer5(en_x4)
        de_h1 = self.deconv1(en_x5)

        h1, h2, w1, w2 = self.resize_shape(en_x4.shape, de_h1.shape)
        h2 = self.layer6(torch.cat([en_x4[:,:,h1:h2,w1:w2], de_h1], dim=1))

        de_h2 = self.deconv2(h2)

        h1, h2, w1, w2 = self.resize_shape(en_x3.shape, de_h2.shape)
        h3 = self.layer7(torch.cat([en_x3[:,:,h1:h2,w1:w2], de_h2], dim=1))

        de_h3 = self.deconv3(h3)

        h1, h2, w1, w2 = self.resize_shape(en_x2.shape, de_h3.shape)
        h4 = self.layer8(torch.cat([en_x2[:,:,h1:h2,w1:w2], de_h3], dim=1))

        de_h4 = self.deconv4(h4)

        h1, h2, w1, w2 = self.resize_shape(en_x1.shape, de_h4.shape)
        h5 = self.layer9(torch.cat([en_x1[:,:,h1:h2,w1:w2], de_h4], dim=1))


        # verify the output shape
        return h5
    def resize_shape(self,shape1, shape2):
        hh1, ww1 = shape1[-2], shape1[-1]
        hh2 ,ww2 = shape2[-2], shape2[-1]
        h1 = int(hh1/2-hh2/2)
        h2 = hh2 + h1
        w1 = int(ww1/2-ww2/2)
        w2 = ww2 + w1
        return h1, h2, w1, w2


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        k = 4  # scaling factor of the network size
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, 64 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(64 // k, 128 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128 // k, 128 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(128 // k, 256 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(256 // k, 256 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer4 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(256 // k, 512 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(512 // k, 512 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer5 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(512 // k, 1024 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(1024 // k, 1024 // k, 3, padding=1), nn.ReLU()
                                    # nn.Conv2d(1024, 512, 2),
                                    )
        self.deconv1 = nn.ConvTranspose2d(1024 // k, 512 // k, 2, stride=2)
        self.layer6 = nn.Sequential(nn.Conv2d(1024 // k, 512 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(512 // k, 512 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv2 = nn.ConvTranspose2d(512 // k, 256 // k, 2, stride=2)
        self.layer7 = nn.Sequential(nn.Conv2d(512 // k, 256 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(256 // k, 256 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv3 = nn.ConvTranspose2d(256 // k, 128 // k, 2, stride=2)
        self.layer8 = nn.Sequential(nn.Conv2d(256 // k, 128 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128 // k, 128 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv4 = nn.ConvTranspose2d(128 // k, 64 // k, 2, stride=2)
        self.layer9 = nn.Sequential(nn.Conv2d(128 // k, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, num_classes, 1)
                                    )

    def forward(self, x):
        en_x1 = self.layer1(x)
        en_x2 = self.layer2(en_x1)
        en_x3 = self.layer3(en_x2)
        en_x4 = self.layer4(en_x3)
        en_x5 = self.layer5(en_x4)
        de_h1 = self.deconv1(en_x5)

        h1, h2, w1, w2 = self.resize_shape(en_x4.shape, de_h1.shape)
        h2 = self.layer6(torch.cat([en_x4[:, :, h1:h2, w1:w2], de_h1], dim=1))

        de_h2 = self.deconv2(h2)

        h1, h2, w1, w2 = self.resize_shape(en_x3.shape, de_h2.shape)
        h3 = self.layer7(torch.cat([en_x3[:, :, h1:h2, w1:w2], de_h2], dim=1))

        de_h3 = self.deconv3(h3)

        h1, h2, w1, w2 = self.resize_shape(en_x2.shape, de_h3.shape)
        h4 = self.layer8(torch.cat([en_x2[:, :, h1:h2, w1:w2], de_h3], dim=1))

        de_h4 = self.deconv4(h4)

        h1, h2, w1, w2 = self.resize_shape(en_x1.shape, de_h4.shape)
        h5 = self.layer9(torch.cat([en_x1[:, :, h1:h2, w1:w2], de_h4], dim=1))

        # verify the output shape
        return h5

    def resize_shape(self, shape1, shape2):
        hh1, ww1 = shape1[-2], shape1[-1]
        hh2, ww2 = shape2[-2], shape2[-1]
        h1 = int(hh1 / 2 - hh2 / 2)
        h2 = hh2 + h1
        w1 = int(ww1 / 2 - ww2 / 2)
        w2 = ww2 + w1
        return h1, h2, w1, w2

class FCN_backbone(nn.Module):
    def __init__(self, num_classes,
                 retrain=True,
                 backbone='resnet101'):
        super(FCN_backbone, self).__init__()

        self.n_class = num_classes
        self.encoder = self.load_encoder(backbone)
        if retrain:
            for params in self.encoder.parameters():
                params.requires_grad = True
        else:
            for params in self.encoder.parameters():
                params.requires_grad = False

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_chnnel[backbone], 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
        )

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
    def forward(self, x):
        x = self.encoder(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        out_decoder = self.deconv5(x)
        score = self.classifier(out_decoder)
        return score

    def load_encoder(self, backbone):
        pretrained_net = pretrained_models[backbone](pretrained=False)
        encoder = nn.Sequential()

        if backbone.startswith('res'):
            for idx, layer in enumerate(pretrained_net.children()):
                # Change the first conv and last linear layer
                if isinstance(layer, nn.Linear) == False and isinstance(layer, nn.AdaptiveAvgPool2d) == False:
                    encoder.add_module(str(idx), layer)
        elif backbone.startswith('vgg'):
            encoder=pretrained_net.features

        return encoder

if __name__ == "__main__":
    model = FCN_backbone(2)
    #x = torch.randn(1,3,572,572)
    x = torch.randn(1,3,512,512)
    print(model)
    print(model(x).shape)
