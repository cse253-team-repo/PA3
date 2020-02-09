from models import UNet
from basic_fcn import FCN
import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
import torch.functional as F
from utils import *
# from tqdm import tqdm

import pdb

CUDA_DIX = [0,1]
class Train:
	def __init__(self,
				 test_path = "./test.csv",
				 train_path = "./train.csv",
				 valid_path = "./val.csv",
				 model="UNet",
				 loss_method = "cross-entropy",
				 opt_method ="Adam",
				 batch_size = 2,
				 img_shape = (512,512),
				 epochs = 1000,
				 num_classes=  34,
				 lr = 0.01,
				 GPU = True,
				 save_best = True,
				 retrain = False,
				 save_path = "my_model.pt"
				):
		self.batch_size = batch_size
		self.epochs = epochs
		self.num_classes = num_classes
		self.lr = lr
		self.opt_method = opt_method
		self.loss_method = loss_method
		self.save_best = save_best
		self.save_path = save_path
		self.retrain = retrain
		if GPU:
			self.gpus = [ix for ix in CUDA_DIX]
		else:
			self.gpus =[]
		self.device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
		self.num_gpus = len(self.gpus)

		networks = {"UNet":UNet,
					"base_fc":FCN}

		self.model = networks[model](num_classes).to(self.device)


		if self.num_gpus > 1:
			self.model = nn.DataParallel(self.model, device_ids=self.gpus)

		self.opt_method = opt_method
		transform = transforms.Compose([
			RandomCrop(img_shape),
			ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406],
					  std=[0.229, 0.224, 0.225])
		])
		self.train_dst = CityScapesDataset(train_path,transforms=transform)
		self.valid_dst = CityScapesDataset(valid_path,transforms=transform)
		self.test_dst = CityScapesDataset(test_path,transforms=transform)
		print("Train set {}\n"
			  "Validation set {}\n"
			  "Test set {}".format(
			len(self.train_dst),
			len(self.valid_dst),
			len(self.test_dst)))
		self.train_loader = DataLoader(self.train_dst,
									   batch_size=batch_size,
									   shuffle=True,
									   num_workers=4)
		self.valid_loader = DataLoader(self.valid_dst,
									   batch_size=batch_size,
									   shuffle=True, num_workers=4)
		self.test_loader = DataLoader(self.test_dst,
									  batch_size=batch_size,
									  shuffle=True,
									  num_workers=4)
		if not self.retrain:
			self.load_weights(self.save_path)

		

		#self.iterations = int(len(self.train_dst) / batch_size)

	def train_on_batch(self, verbose=True, lr_decay=True):

		if self.opt_method == "Adam":
			optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
			if lr_decay:
				lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		if self.loss_method == "cross-entropy":
			criterio = nn.CrossEntropyLoss().to(self.device)
		loss_epoch = []
		valid_accs = []
		MAX = 0
		for epoch in range(self.epochs):
			loss_itr = []
			for i, (img, target, label) in enumerate(self.train_loader):
				itr_start = time.time()
				#print(train_x.shape)
				#pdb.set_trace()
				img = img.to(self.device)
				train_y_one_hot = target.to(self.device)
				train_y = label.to(self.device)

				optimizer.zero_grad()
				output = self.model(img)
				# print(train_y_one_hot.shape,output.shape)
				loss = criterio(output, train_y)

				loss.backward()
				optimizer.step()
				loss_itr.append(loss.item())
				itr_end = time.time()
				if verbose:
					print("Iterations: {} \t training loss: {} \t time: {}".format(i, loss_itr[-1], itr_end - itr_start))
			loss_epoch.append(np.mean(loss_itr))
			print("*"*10)
			print("Epoch: {} \t training loss: {}".format(epoch, loss_epoch[-1]))
			if lr_decay:
				lr_sheduler.step(epoch)

			valid_acc, valid_loss = self.check_accuracy(self.valid_loader, get_loss=True)
			print("Epoch: {} \t valid loss: {} \t valid accuracy: {}".format(epoch, valid_loss, valid_acc))
			if self.save_best:
				if valid_acc > MAX:
					self.save_weights(self.save_path)
					MAX = valid_acc
			valid_accs.append(valid_acc)


	def check_accuracy(self, dataloader, get_loss=True):
		accs = []
		losses = []
		if self.loss_method == "cross-entropy":
			criterio = nn.CrossEntropyLoss()
		with torch.no_grad():
			for i, data in enumerate(dataloader):
				x, y_one_hot, y = data
				x = x.to(self.device)
				y_one_hot = y_one_hot.to(self.device)
				y = y.to(self.device)
				out = self.model(x)
				loss = criterio(out, y)
				losses.append(loss.cpu().numpy())
				y_hat = torch.argmax(out,dim=1)
				acc = pixel_acc(y_hat, y)
				accs.append(acc)
		if get_loss:
			return np.mean(accs), np.mean(losses)
		return np.mean(accs)

	def save_weights(self,path):
		print("Saving the model ...")
		torch.save(self.model.state_dict(), path)
		print("Saving Done!")

	def load_weights(self,path):
		print("Loading the parameters")
		self.model.load_state_dict(torch.load(path))
		self.model.eval()



if __name__ == "__main__":
	train = Train()
	train.train_on_batch()


