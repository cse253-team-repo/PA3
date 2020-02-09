import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(64),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(256),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
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
        # x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        out_encoder = self.conv5(x)
        # score = __(self.relu(__(out_encoder)))     
        # Complete the forward function for the rest of the decoder
        y = self.deconv1(out_encoder)
        y = self.deconv2(y)
        y = self.deconv3(y)
        y = self.deconv4(y)
        out_decoder = self.deconv5(y)

        score = self.classifier(out_decoder)                   
        print("score shape: ", score.shape)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)