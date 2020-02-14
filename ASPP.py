import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class ASPP(nn.Module):
	def __init__(self,rates=[6,12,18], in_channels=64, out_channel=128, pooling_method="average_pooling"):
		self.branch = []
		self.apppend(nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=1,padding=0, dilation=1))
		for i in range(len(rates)):
			self.branch.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=rates[i], dilation=rates[i]))
		if pooling_method == "average_pooling":
			self.pooling = nn.sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
										nn.conv2d(in_channel, out_channel, 1, stride=1, padding=0, dilation=1),
										nn.BatchNorm2d(in_channel),
										nn.ReLU(inplace=True)
										)
		else:
			raise ValueError(f"Not implement the pooling metho {pooling_method}")

	def forward(self, x):
		branch_output = []
		for layer in self.branch:
			branch_output.append(layer(x))

		x_conv = torch.cat(branch_output+[self.pooling(x)], dim=1)
		return x_conv





if __name__ == "__main__":
	x = torch.randn(2,3,512, 1024)
	model = ASPP(in_channel=3, out_channel=256)
	y = model(x)
	print(y.shape)
