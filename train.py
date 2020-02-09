from models import UNet
from basic_fcn import FCN
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
from utils import *
from utils import load_config
import yaml
# from tqdm import tqdm

import pdb

CUDA_DIX = [0,1,2,3]
class Train:
<<<<<<< HEAD
	def __init__(self, test_path="./test.csv", train_path = "./train.csv", valid_path = "./val.csv", 
					transform='resize', model="UNet", loss_method="cross-entropy", opt_method ="Adam",
					batch_size=12, img_shape=(512,512), epochs=1000, num_classes=34, lr=0.01, 
					GPU=True
=======
	def __init__(self,
				 config,
				 test_path = "./test.csv",
				 train_path = "./train.csv",
				 valid_path = "./val.csv",
				 save_path = "my_model.pt"
>>>>>>> refs/remotes/origin/master
				):
		self.save_path = save_path
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
		self.device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
		self.num_gpus = len(self.gpus)

		networks = {"UNet":UNet,
					"base_fc":FCN}

		self.model = networks[model](self.num_classes).to(self.device)


		if self.num_gpus > 1:
			self.model = nn.DataParallel(self.model, device_ids=self.gpus)

		transform = transforms.Compose([
			RandomResizedCrop(img_shape),
			ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406],
					  std=[0.229, 0.224, 0.225])
		])
		test_transform = transforms.Compose([
			ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406],
					  std=[0.229, 0.224, 0.225])
		])
		self.train_dst = CityScapesDataset(train_path,transforms=transform)
		self.valid_dst = CityScapesDataset(valid_path,transforms=test_transform)
		self.test_dst = CityScapesDataset(test_path,transforms=test_transform)
		print("Train set {}\n"
			  "Validation set {}\n"
			  "Test set {}".format(
			len(self.train_dst),
			len(self.valid_dst),
			len(self.test_dst)))
		self.train_loader = DataLoader(self.train_dst,
									   batch_size=self.batch_size,
									   shuffle=True,
									   num_workers=4)
		self.valid_loader = DataLoader(self.valid_dst,
									   batch_size=2,
									   shuffle=True, num_workers=4)
		self.test_loader = DataLoader(self.test_dst,
									  batch_size=2,
									  shuffle=True,
									  num_workers=4)
		if self.retrain == True:
			self.load_weights(self.save_path)



		#self.iterations = int(len(self.train_dst) / batch_size)

<<<<<<< HEAD
	def train_on_batch(self, verbose=False):
=======
	def train_on_batch(self, verbose=True, lr_decay=True):

>>>>>>> refs/remotes/origin/master
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
			print("Epoch: {} \t training loss: {}".format(epoch, np.mean(loss_epoch)))
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
	config = load_config("Unet_config.yaml")
	train = Train(config)
	train.train_on_batch()


