import torch
import torch.utils.data as Data
import audioDataset
import random
# simple test to debug for audioDataset.py
batch_size=64
epochNum=5
audioDataSet=audioDataset.AudioDataset4speakerClassify('../dataprocessing/train_file_path.txt')
loader=Data.DataLoader(
		dataset=audioDataSet,
		batch_size=batch_size,
		shuffle=True,
		num_workers=2
	)
print (len(audioDataSet))

for epoch in range(epochNum):
	print ('**********EPOCH',epoch,'*************')
	for step,(x1,y) in enumerate(loader):
		print (x1.shape,y.shape)

