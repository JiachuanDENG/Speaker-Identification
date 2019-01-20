import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EncoderNetClassifier(nn.Module):
	def __init__(self,speakersNum=19):
		super(EncoderNetClassifier, self).__init__()

		################
		### Method 1 ###
		################
		self.speakersNum=speakersNum
		self.conv11 = nn.Conv3d(1, 8, (3, 7, 7), stride=(1, 2, 1))
		self.conv11_bn = nn.BatchNorm3d(8)
		self.conv11_activation = torch.nn.PReLU()
		self.conv12 = nn.Conv3d(8, 8, (3, 7, 7), stride=(1, 1, 1))
		self.conv12_bn = nn.BatchNorm3d(8)
		self.conv12_activation = torch.nn.PReLU()
		self.conv21 = nn.Conv3d(8, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv21_bn = nn.BatchNorm3d(16)
		self.conv21_activation = torch.nn.PReLU()
		self.conv22 = nn.Conv3d(16, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv22_bn = nn.BatchNorm3d(16)
		self.conv22_activation = torch.nn.PReLU()
		self.conv31 = nn.Conv3d(16, 32, (2, 3, 3), stride=(1, 1, 1))
		self.conv31_bn = nn.BatchNorm3d(32)
		self.conv31_activation = torch.nn.PReLU()
		# self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
		# self.conv32_bn = nn.BatchNorm3d(64)
		# self.conv32_activation = torch.nn.PReLU()
		# self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
		# self.conv41_bn = nn.BatchNorm3d(128)
		# self.conv41_activation = torch.nn.PReLU()



		# Fully-connected
		self.fc1 = nn.Linear(32 * 3* 1 * 18, 128) # input size (Batchsize,1,10,40,40)
		self.fc1_bn = nn.BatchNorm1d(128)
		self.fc1_activation = torch.nn.PReLU()
		self.fc2 = nn.Linear(128, speakersNum)


	def forward(self, x):
		# Method-1
		x = self.conv11_activation(self.conv11_bn(self.conv11(x)))
		x = self.conv12_activation(self.conv12_bn(self.conv12(x)))
		x = self.conv21_activation(self.conv21_bn(self.conv21(x)))
		x = self.conv22_activation(self.conv22_bn(self.conv22(x)))
		x = self.conv31_activation(self.conv31_bn(self.conv31(x)))
		# x = self.conv32_activation(self.conv32_bn(self.conv32(x)))
		# x = self.conv41_activation(self.conv41_bn(self.conv41(x)))
		# print (x.shape)
		x = x.view(-1, 32*3*1*18)
		x = self.fc1_activation(self.fc1_bn(self.fc1(x)))
		x=self.fc2(x)
		# x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
		return x
class EncoderNet(nn.Module):
	def __init__(self,embedding_size):
		super(EncoderNet, self).__init__()

		################
		### Method 1 ###
		################
		self.embedding_size=embedding_size
		self.conv11 = nn.Conv3d(1, 8, (3, 7, 7), stride=(1, 2, 1))
		self.conv11_bn = nn.BatchNorm3d(8)
		self.conv11_activation = torch.nn.PReLU()
		self.conv12 = nn.Conv3d(8, 8, (3, 7, 7), stride=(1, 1, 1))
		self.conv12_bn = nn.BatchNorm3d(8)
		self.conv12_activation = torch.nn.PReLU()
		self.conv21 = nn.Conv3d(8, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv21_bn = nn.BatchNorm3d(16)
		self.conv21_activation = torch.nn.PReLU()
		self.conv22 = nn.Conv3d(16, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv22_bn = nn.BatchNorm3d(16)
		self.conv22_activation = torch.nn.PReLU()
		self.conv31 = nn.Conv3d(16, 32, (2, 3, 3), stride=(1, 1, 1))
		self.conv31_bn = nn.BatchNorm3d(32)
		self.conv31_activation = torch.nn.PReLU()
		# self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
		# self.conv32_bn = nn.BatchNorm3d(64)
		# self.conv32_activation = torch.nn.PReLU()
		# self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
		# self.conv41_bn = nn.BatchNorm3d(128)
		# self.conv41_activation = torch.nn.PReLU()



		# Fully-connected
		self.fc1 = nn.Linear(32 * 3* 1 * 18, 128) # input size (Batchsize,1,10,40,40)
		self.fc1_bn = nn.BatchNorm1d(128)
		self.fc1_activation = torch.nn.PReLU()
		self.fc2 = nn.Linear(128, embedding_size)


	def forward(self, x):
		# Method-1
		x = self.conv11_activation(self.conv11_bn(self.conv11(x)))
		x = self.conv12_activation(self.conv12_bn(self.conv12(x)))
		x = self.conv21_activation(self.conv21_bn(self.conv21(x)))
		x = self.conv22_activation(self.conv22_bn(self.conv22(x)))
		x = self.conv31_activation(self.conv31_bn(self.conv31(x)))
		# x = self.conv32_activation(self.conv32_bn(self.conv32(x)))
		# x = self.conv41_activation(self.conv41_bn(self.conv41(x)))
		# print (x.shape)
		x = x.view(-1, 32*3*1*18)
		x = self.fc1_activation(self.fc1_bn(self.fc1(x)))
		x=self.fc2(x)
		x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
		return x

class SiameseNet(nn.Module):
	def __init__(self,embedding_size,distance_metrics='l1'):
		super(SiameseNet,self).__init__()
		print ('new version')
		self.embedding_size=embedding_size
		self.distance_metrics=distance_metrics
		# self.encoder=EncoderNet(self.embedding_size)

		
		self.conv11 = nn.Conv3d(1, 8, (3, 7, 7), stride=(1, 2, 1))
		self.conv11_bn = nn.BatchNorm3d(8)
		self.conv11_activation = torch.nn.PReLU()
		self.conv12 = nn.Conv3d(8, 8, (3, 7, 7), stride=(1, 1, 1))
		self.conv12_bn = nn.BatchNorm3d(8)
		self.conv12_activation = torch.nn.PReLU()
		self.conv21 = nn.Conv3d(8, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv21_bn = nn.BatchNorm3d(16)
		self.conv21_activation = torch.nn.PReLU()
		self.conv22 = nn.Conv3d(16, 16, (2, 5, 5), stride=(1, 1, 1))
		self.conv22_bn = nn.BatchNorm3d(16)
		self.conv22_activation = torch.nn.PReLU()
		self.conv31 = nn.Conv3d(16, 32, (2, 3, 3), stride=(1, 1, 1))
		self.conv31_bn = nn.BatchNorm3d(32)
		self.conv31_activation = torch.nn.PReLU()
		# self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
		# self.conv32_bn = nn.BatchNorm3d(64)
		# self.conv32_activation = torch.nn.PReLU()
		# self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
		# self.conv41_bn = nn.BatchNorm3d(128)
		# self.conv41_activation = torch.nn.PReLU()



		# Fully-connected
		self.fc1 = nn.Linear(32 * 3* 1 * 18, 128) # input size (Batchsize,1,10,40,40)
		self.fc1_bn = nn.BatchNorm1d(128)
		self.fc1_activation = torch.nn.PReLU()
		self.fc2 = nn.Linear(128, embedding_size)


		self.distanceActFunc=nn.Sigmoid()
		self.FC=nn.Linear(self.embedding_size,2)

	def getfeature(self, x):
		# Method-1
		x = self.conv11_activation(self.conv11_bn(self.conv11(x)))
		x = self.conv12_activation(self.conv12_bn(self.conv12(x)))
		x = self.conv21_activation(self.conv21_bn(self.conv21(x)))
		x = self.conv22_activation(self.conv22_bn(self.conv22(x)))
		x = self.conv31_activation(self.conv31_bn(self.conv31(x)))
		# x = self.conv32_activation(self.conv32_bn(self.conv32(x)))
		# x = self.conv41_activation(self.conv41_bn(self.conv41(x)))
		# print (x.shape)
		x = x.view(-1, 32*3*1*18)
		x = self.fc1_activation(self.fc1_bn(self.fc1(x)))
		x=self.fc2(x)
		x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
		return x

	def forward(self,x1,x2):
		emb1,emb2=self.getfeature(x1),self.getfeature(x2)
		if self.distance_metrics=='l1':
			distance=torch.abs(emb1-emb2)
		elif self.distance_metrics =='cosine':
			raise NotImplementedError
		else:
			raise NotImplementedError
		
		out=self.FC(distance)
		return out











    