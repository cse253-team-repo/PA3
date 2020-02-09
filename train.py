from models import UNet
import torch
import torch.nn as nn
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class Train:
	def __init__(self, batch_size, epochs=1, method="cross-entropy", num_classes=10):
		model = UNet(num_classes).to(device)
		x = torch.zeros(batch_size, 3, 512, 1024, device=device)
		y =	torch.empty(batch_size, 512, 1024, dtype=torch.long).random_(num_classes).to(device)
		out = model(x)
		if method == "cross-entropy":
			criterio = nn.CrossEntropyLoss()
		loss = criterio(out, y)
		loss.backward()
		loss = loss.detach()
		print(loss)

if __name__ == "__main__":
	train = Train(1)

