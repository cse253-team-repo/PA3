from models import UNet
import torch
import torch.nn as nn
from dataloader import CityScapesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
import pdb




class Train:
	def __init__(self, test_path="./test.csv", train_path = "./train.csv", valid_path = "./val.csv",
					model="UNet", loss_method="cross-entropy", opt_method ="Adam",
					batch_size=1, img_shape=(512,512), epochs=1000, num_classes=32, lr=0.01, 
					GPU=True
				):
		self.batch_size = batch_size
		self.epochs = epochs
		self.num_classes = num_classes
		self.lr = lr
		self.opt_method = opt_method
		self.loss_method = loss_method
		if GPU:
			self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cpu")

		if model == "UNet":
			self.model = UNet(num_classes).to(self.device)
		else:
			raise ValueError("Not implement {}".format(model))
		self.opt_method = opt_method
		self.train_dst = CityScapesDataset(train_path)
		self.valid_dst = CityScapesDataset(valid_path)
		self.test_dst = CityScapesDataset(test_path)

		self.train_loader = DataLoader(self.train_dst, batch_size=batch_size, shuffle=True, num_workers=8)
		self.valid_loader = DataLoader(self.valid_dst, batch_size=batch_size, shuffle=True, num_workers=8)
		self.test_loader = DataLoader(self.test_dst, batch_size=batch_size, shuffle=True, num_workers=8)
		

		#self.iterations = int(len(self.train_dst) / batch_size)

	def train_on_batch(self, verbose=True):
		if self.opt_method == "Adam":
			optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		if self.loss_method == "cross-entropy":
			criterio = nn.CrossEntropyLoss()
		loss_epoch = []
		for epoch in range(self.epochs):
			loss_itr = []
			for i, data in enumerate(self.train_loader):
				itr_start = time.time()
				optimizer.zero_grad()
				train_x, train_y_one_hot, train_y = data
				#print(train_x.shape)
				#pdb.set_trace()
				train_x = train_x.to(self.device)
				train_y_one_hot = train_y_one_hot.to(self.device)
				train_y = train_y.to(self.device)
				output = self.model(train_x)
				loss = criterio(output, train_y)
				loss.backward()
				optimizer.step()
				loss = loss.cpu().detach().numpy()
				loss_itr.append(loss)
				itr_end = time.time()
				if verbose:
					print("Iterations: {} \t loss: {} \t time: {}".format(i, loss_itr[-1], itr_start - itr_end))
			loss_epoch.append(np.mean(loss_itr))
			print("*"*10)
			print("Epoch: {} \t loss: {}".format(epoch, loss_epoch[-1]))








if __name__ == "__main__":
	train = Train()
	x = torch.randn(1, 3, 512, 512)
	y =	torch.empty(1, 512, 512, dtype=torch.long).random_(32)
	train.train_loader = [(x, y, y)]
	train.train_on_batch()


