import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb


class ASPP(nn.Module):
	def __init__(self,rates=[6,12,18], in_channel=64, out_channel=128, pooling_method="average_pooling", p=0):
		"""
			rates: dilation rate
			in_channel: number of input channels
			out_channel: number of output channels
			pooling_method: default is average pooling
			p: dropout rate
		"""
		super(ASPP,self).__init__()
		self.branch = []
		self.branch.append(nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=1,padding=0, dilation=1))
		for i in range(len(rates)):
			self.branch.append(nn.Sequential(
								nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=rates[i], dilation=rates[i]),
								nn.Dropout(p)
								)
							)
		if pooling_method == "average_pooling":
			self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
										nn.Conv2d(in_channel, out_channel,  kernel_size=(1,1), stride=1, padding=0, dilation=1),
										nn.BatchNorm2d(out_channel),
										nn.ReLU(inplace=True),
										)
		else:
			raise ValueError(f"Not implement the pooling metho {pooling_method}")

	def forward(self, x):
		"""
			compute convolution with different scaling size and pooling in parrallel architecture and concatenate them together
		"""
		branch_output = []
		for layer in self.branch:
			branch_output.append(layer(x))
		branch_output.append(F.interpolate(self.pooling(x), size=x.shape[2:], mode='bilinear', align_corners=True))
		x_conv = torch.cat(branch_output, dim=1)
		return x_conv


class Deeplabv3(nn.Module):
	def __init__(self)



if __name__ == "__main__":
	x = torch.randn(2,3,256, 512)
	model = ASPP(in_channel=3, out_channel=64)
	y = model(x)
	print(y.shape)
