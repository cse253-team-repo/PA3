import torch
import torch.nn as nn
import torch.nn.functional as F
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





class UNet(nn.Module):
	def __init__(self, num_classes):
		super(UNet, self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
									nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
									)
		self.layer2 = nn.Sequential(nn.MaxPool2d(2), 
									nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
									nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
									)
		self.layer3 = nn.Sequential(nn.MaxPool2d(2),
									nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
									nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
									)
		self.layer4 = nn.Sequential(nn.MaxPool2d(2),
									nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
									nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
									)
		self.layer5 = nn.Sequential(nn.MaxPool2d(2),
									nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(),
									nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU()
									#nn.Conv2d(1024, 512, 2), 
									)
		self.deconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
		self.layer6 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(),
									nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
									)
		self.deconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.layer7 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
									nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
									)
		self.deconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.layer8 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
									nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
									)
		self.deconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.layer9 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
									nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
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



if __name__ == "__main__":
	model = UNet(2)
	#x = torch.randn(1,3,572,572)
	x = torch.randn(1,3,1024,2048)
	model(x)
	#print(model(x).shape)