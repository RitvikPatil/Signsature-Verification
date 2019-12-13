from torch.nn import Linear, Conv2d, MaxPool2d, LocalResponseNorm, Dropout
from torch.nn.functional import relu
from torch.nn import Module
from PIL import Image
from PIL.ImageOps import invert
import numpy as np
from torch.tensor import Tensor
from torch.utils.data import Dataset
from random import randrange
from sklearn.model_selection import train_test_split
import pickle
from torch import save
from torch.optim import Adam
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

# Preprocessing and Dataloaders


def invert_image(path):
	image_file = Image.open(path) # open colour image
	image_file = image_file.convert('L').resize([220, 155])
	image_file = invert(image_file)
	image_array = np.array(image_file)
	for i in range(image_array.shape[0]):
		for j in range(image_array.shape[1]):
			if image_array[i][j]<=50:
				image_array[i][j]=0
			else:
				image_array[i][j]=255
	return image_array

def convert_to_image_tensor(image_array):
	image_array = image_array/255.0
	return Tensor(image_array).view(1, 220, 155)

base_path_org = 'CEDAR signature verification/full_org/original_%d_%d.png'
base_path_forg = 'CEDAR signature verification/full_forg/forgeries_%d_%d.png'

def fix_pair(x, y):
	if x == y:
		return fix_pair(x, randrange(1, 24))
	else:
		return x, y

data = []
n_samples_of_each_class = 900

prefix ='/content/drive/My Drive/'

for _ in range(n_samples_of_each_class):
	anchor_person = randrange(1, 55)
	anchor_sign = randrange(1, 24)
	pos_sign = randrange(1, 24)
	anchor_sign, pos_sign = fix_pair(anchor_sign, pos_sign)
	neg_sign = randrange(1, 24)
	positive = [base_path_org%(anchor_person, anchor_sign), base_path_org%(anchor_person, pos_sign), 1]
	negative = [base_path_org%(anchor_person, anchor_sign), base_path_forg%(anchor_person, neg_sign), 0]
	data.append(positive)
	data.append(negative)


train, test = train_test_split(data, test_size=0.15)
with open('train_index.pkl', 'wb') as train_index_file:
	pickle.dump(train, train_index_file)

with open('test_index.pkl', 'wb') as test_index_file:
	pickle.dump(test, test_index_file)


class TrainDataset(Dataset):

	def __init__(self):
		with open('train_index.pkl', 'rb') as train_index_file:
			self.pairs = pickle.load(train_index_file)

	def __getitem__(self, index):
		item = self.pairs[index]
		X = convert_to_image_tensor(invert_image(prefix+item[0]))
		Y = convert_to_image_tensor(invert_image(prefix+item[1]))
		return [X, Y, item[2]]

	def __len__(self):
		return len(self.pairs)


class TestDataset(Dataset):

	def __init__(self):
		with open('test_index.pkl', 'rb') as test_index_file:
			self.pairs = pickle.load(test_index_file)

	def __getitem__(self, index):
		item = self.pairs[index]
		X = convert_to_image_tensor(invert_image(prefix+item[0]))
		Y = convert_to_image_tensor(invert_image(prefix+item[1]))
		return [X, Y, item[2]]

	def __len__(self):
		return len(self.pairs)
  

class SiameseConvNet(Module):
	def __init__(self):
		super().__init__()
		self.conv1 = Conv2d(1, 48, kernel_size=(11, 11), stride=1)
		self.lrn1 = LocalResponseNorm(48, alpha=1e-4, beta=0.75, k=2)
		self.pool1 = MaxPool2d(kernel_size=(3, 3), stride=2)
		self.conv2 = Conv2d(48, 128, kernel_size=(5, 5), stride=1, padding=2)
		self.lrn2 = LocalResponseNorm(128, alpha=1e-4, beta=0.75, k=2)
		self.pool2 = MaxPool2d(kernel_size=(3, 3), stride=2)
		self.dropout1 = Dropout(0.3)
		self.conv3 = Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
		self.conv4 = Conv2d(256, 96, kernel_size=(3, 3), stride=1, padding=1)
		self.pool3 = MaxPool2d(kernel_size=(3,3), stride=2)
		self.dropout2 = Dropout(0.3)
		self.fc1 = Linear(25 * 17 * 96, 1024)
		self.dropout3 = Dropout(0.5)
		self.fc2 = Linear(1024, 128)

	def forward_once(self, x):
		x = relu(self.conv1(x))
		x = self.lrn1(x)
		x = self.pool1(x)
		x = relu(self.conv2(x))
		x = self.lrn2(x)
		x = self.pool2(x)
		x = self.dropout1(x)
		x = relu(self.conv3(x))
		x = relu(self.conv4(x))
		x = self.pool3(x)
		x = self.dropout2(x)
		x = x.view(-1, 17 * 25 * 96)
		x = relu(self.fc1(x))
		x = self.dropout3(x)
		x = relu(self.fc2(x))
		return x

	def forward(self, x, y):
		f_x = self.forward_once(x)
		f_y = self.forward_once(y)
		return f_x, f_y


def distance_metric(features_A, features_B):
	batch_losses = F.pairwise_distance(features_A, features_B)
	return batch_losses


class ContrastiveLoss(torch.nn.Module):

	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2)
		loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive


model = SiameseConvNet().cuda()
criterion = ContrastiveLoss().cuda()
optimizer = Adam(model.parameters())

train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=False)


def checkpoint(epoch):
	file_path = "model_epoch_%d" % epoch
	with open(file_path, 'wb') as f:
		save(model.state_dict(), f)


def train(epoch):
	for batch_index, data in enumerate(train_loader):
		A = data[0].cuda()
		B = data[1].cuda()
		optimizer.zero_grad()
		label = data[2].float().cuda()
		f_A, f_B = model.forward(A, B)
		loss = criterion(f_A, f_B, label)
		print('Epoch {}, batch {}, loss={}'.format(epoch, batch_index, loss.item()))
		loss.backward()
		optimizer.step()


for e in range(1, 21):
	train(e)
	checkpoint(e)

