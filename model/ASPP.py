import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import *
import pdb
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

class ASPP(nn.Module):
	def __init__(self, in_channel, out_channel, h_channel=128, rates=[6,12,18], pooling_method="average_pooling", p=0):
		"""
			rates: dilation rate
			in_channel: number of input channels
			h_channel: number of hidden channel for each convolution branch
			out_channel: number of output channels
			pooling_method: default is average pooling
			p: dropout rate
		"""
		super(ASPP,self).__init__()
		self.branch = nn.ModuleList([])
		self.branch.append(nn.Conv2d(in_channel, h_channel, kernel_size=(1,1), stride=1,padding=0, dilation=1))
		for i in range(len(rates)):
			self.branch.append(nn.Sequential(
								nn.Conv2d(in_channel, h_channel, kernel_size=(3,3), stride=1, padding=rates[i], dilation=rates[i]),
								nn.BatchNorm2d(h_channel),
								nn.ReLU(inplace=True),
								nn.Dropout(p)
								)
							)
		if pooling_method == "average_pooling":
			self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
										nn.Conv2d(in_channel, h_channel,  kernel_size=(1,1), stride=1, padding=0, dilation=1),
										nn.BatchNorm2d(h_channel),
										nn.ReLU(inplace=True),
										)
		else:
			raise ValueError(f"Not implement the pooling method {pooling_method}")
		self.encoder = nn.Sequential(nn.Conv2d(h_channel*(len(rates)+1+1), out_channel, kernel_size=(1,1), stride=1, padding=0, dilation=1),
									nn.BatchNorm2d(out_channel),
									nn.ReLU(inplace=True)
									)

	def forward(self, x):
		"""
			compute convolution with different scaling size and pooling in parrallel architecture and concatenate them together
		"""
		branch_output = []
		for layer in self.branch:
			branch_output.append(layer(x))
		branch_output.append(F.interpolate(self.pooling(x), size=x.shape[2:], mode='bilinear', align_corners=True))
		x = torch.cat(branch_output, dim=1)
		x = self.encoder(x)
		return x


class BasicModel(nn.Module):
	def __init__(self, num_classes,
					use_torch_model=False,
					retrain_backbone=True,
					backbone='resnet50'):
		super(BasicModel,self).__init__()
		self.num_classes = num_classes
		self.retrain_backbone= retrain_backbone
		self.backbone = backbone
		self.use_torch_model = use_torch_model

	def load_encoder(self, backbone):
		pretrained_net = pretrained_models[backbone](pretrained=True)
		encoder = nn.Sequential()

		if backbone.startswith('res'):
			for idx, layer in enumerate(pretrained_net.children()):
				# Change the first conv and last linear layer
				if isinstance(layer, nn.Linear) == False and isinstance(layer, nn.AdaptiveAvgPool2d) == False:
					encoder.add_module(str(idx), layer)
		elif backbone.startswith('vgg'):
			encoder=pretrained_net.features

		return encoder
	def resize_shape(self, shape1, shape2):
		hh1, ww1 = shape1[-2], shape1[-1]
		hh2, ww2 = shape2[-2], shape2[-1]
		h1 = int(hh1 / 2 - hh2 / 2)
		h2 = hh2 + h1
		w1 = int(ww1 / 2 - ww2 / 2)
		w2 = ww2 + w1
		return h1, h2, w1, w2
	def make_mlp_down(self, in_channel, out_channel):
		return nn.Sequential(
					nn.MaxPool2d(2),
					nn.Conv2d(in_channel, out_channel, 3, padding=1),
					nn.BatchNorm2d(out_channel),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channel, out_channel, 3, padding=1),
					nn.BatchNorm2d(out_channel),
					nn.ReLU(inplace=True)
					)
	def make_mlp_upsample(self, in_channel, out_channel):
		return nn.Sequential(
					nn.Conv2d(in_channel, out_channel, 3, padding=1),
					nn.BatchNorm2d(out_channel),
					nn.ReLU(inplace=True),
					nn.Conv2d(out_channel, out_channel, 3, padding=1),
					nn.BatchNorm2d(out_channel),
					nn.ReLU(inplace=True)
					)
	def make_decoder(self, in_channel, out_channel, mode='bilinear', scale_factor=2):
		return nn.Sequential(
						nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
						nn.Conv2d(in_channel, out_channel, 3, padding=1),
						nn.BatchNorm2d(out_channel),
						nn.ReLU(inplace=True)
						)




class Deeplab(BasicModel):
	def __init__(self, num_classes,
				use_torch_model=False,
				retrain_backbone=True,
				backbone='resnet50'):
		super(Deeplab, self).__init__(num_classes, use_torch_model, retrain_backbone, backbone)
		if use_torch_model:
			self.encoder = self.load_encoder(backbone)
			if retrain_backbone:
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
			self.aspp = ASPP(32, 32, h_channel=32)
			self.classifier = nn.Conv2d(32, self.num_classes, kernel_size=1)


		else:
			self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
										nn.Conv2d(64, 64, 3, padding=1),
										nn.BatchNorm2d(64),
										nn.ReLU(inplace=True)
										)
			self.enc2 = self.make_mlp_down(64, 128)
			self.enc3 = self.make_mlp_down(128, 256)
			self.enc4 = self.make_mlp_down(256, 512)
			self.enc5 = self.make_mlp_down(512, 1024)
			self.dec1 = self.make_decoder(1024, 512, scale_factor=2)
			self.declayer1 = self.make_mlp_upsample(1024, 512,)
			self.dec2 = self.make_decoder(512, 256, scale_factor=2)
			self.declayer2 = self.make_mlp_upsample(512, 256)
			self.dec3 = self.make_decoder(256, 128, scale_factor=2)
			self.declayer3 = ASPP(256, 128, h_channel=128)
			self.dec4 = self.make_decoder(128, 64, scale_factor=2)
			self.classifier = ASPP(128, self.num_classes, h_channel=self.num_classes)


	def forward(self, x):
		if self.use_torch_model:
			# use pretrained resnet as backbone
			x = self.encoder(x)
			x = self.deconv1(x)
			x = self.deconv2(x)
			x = self.deconv3(x)
			x = self.deconv4(x)
			x = self.deconv5(x)
			#pdb.set_trace()
			x =  self.aspp(x)
			out = self.classifier(x)
		else:
			# skip connection
			x1 = self.enc1(x)
			x2 = self.enc2(x1)
			x3 = self.enc3(x2)
			x4 = self.enc4(x3)
			x5 = self.enc5(x4)
			x6 = self.dec1(x5)
			x7 = self.declayer1(torch.cat([x4, x6], dim=1))

			x8 = self.dec2(x7)
			x9 = self.declayer2(torch.cat([x3, x8], dim=1))
			x10 = self.dec3(x9)
			x11 = self.declayer3(torch.cat([x2, x10], dim=1))
			x12 = self.dec4(x11)
			out = self.classifier(torch.cat([x1, x12], dim=1))
		return out


class Deeplab_yxy(BasicModel):
	def __init__(self, num_classes,
				use_torch_model=True,
				retrain_backbone=True,
				backbone='resnet50'):
		super(Deeplab_yxy, self).__init__(num_classes, use_torch_model, retrain_backbone, backbone)
		if use_torch_model:
			self.encoder = self.load_encoder(backbone)
			if retrain_backbone:
				for params in self.encoder.parameters():
					params.requires_grad = True
			else:
				for params in self.encoder.parameters():
					params.requires_grad = False

			self.aspp = ASPP(32, 32, h_channel=32)
			self.classifier = nn.Conv2d(32, self.num_classes, kernel_size=1)


	def forward(self, x):
		input_shape = x.shape
		x = self.encoder(x)
		x =  self.aspp(x)
		x = self.classifier(x)
		out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		return out



if __name__ == "__main__":
	x = torch.randn(2,3,256, 512)
	model = Deeplab(12, use_torch_model=True)
	print(model)
	y = model(x)
	print(y.shape)
