from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import configparser
from utils import audioDataset
import torch.utils.data as Data
from  model import models
import torch
import random
import time
import math
import pdb
import sys
import os
import scipy.io as sio
from sklearn import *
from tqdm import tqdm
import numpy as np

BATCH_SIZE=32
def evaluation(encoder,loader,saved_encoder='./savedEncoder.pkl'):
	encoder.load_state_dict(torch.load(saved_encoder))
	score_vector=[]
	target_label_vector=[]
	for step,(x1,x2,y) in tqdm(enumerate(loader)):
		f1=encoder.getFeature(x1).data.numpy()
		f2=encoder.getFeature(x2).data.numpy()
		labels=y.data.tolist()
		for i in range(f1.shape[0]):
			feature1 = Normalizer(norm='l2').fit_transform(f1[i:i+1,:])
			feature2 = Normalizer(norm='l2').fit_transform(f2[i:i+1,:])
			
			score = cosine_similarity(feature1, feature2)
			
			score_vector.append(score)
			target_label_vector.append(labels[i])
	score_vector=np.array(score_vector).reshape([-1,1])
	target_label_vector=np.array(target_label_vector).reshape([-1,1])

	np.save('./score_vector.npy',score_vector)
	np.save('./target_label_vector.npy',target_label_vector)






def calculate_eer_auc_ap(label,distance):

    fpr, tpr, thresholds = metrics.roc_curve(label, distance, pos_label=1)
    AUC = metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    AP = metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x

    return EER,AUC,AP,fpr, tpr

def getEER_AUC():
	score = np.load('./score_vector.npy')
	label = np.load('./target_label_vector.npy')

	# K-fold validation for ROC
	k=1
	step = int(label.shape[0] / float(k))
	EER_VECTOR = np.zeros((k,1))
	AUC_VECTOR = np.zeros((k,1))
	for split_num in range(k):
	    index_start = split_num * step
	    index_end = (split_num + 1) * step
	    EER_temp,AUC_temp,AP,fpr, tpr = calculate_eer_auc_ap(label[index_start:index_end],score[index_start:index_end])
	    EER_VECTOR[split_num] = EER_temp * 100
	    AUC_VECTOR[split_num] = AUC_temp * 100

	print("EER=",np.mean(EER_VECTOR),np.std(EER_VECTOR))
	print("AUC=",np.mean(AUC_VECTOR),np.std(AUC_VECTOR))


	

def main():
	config=configparser.ConfigParser()
	config.read('./model/model_config.ini')
	embedding_size=int(config.get('ENCODER','embedding_size'))
	speaker_num=int(config.get('ENCODER','speaker_num'))

	audioDataSetVal=audioDataset.AudioDataset4Siamese(siameseValFilePath)
	
	loaderVal=Data.DataLoader(
	dataset=audioDataSetVal,
	batch_size=BATCH_SIZE,
	shuffle=True
	)
	
	encoderNet=models.EncoderNetClassifier(embedding_size=embedding_size,speakersNum=speaker_num)
	print ('evaluation...')
	evaluation(encoderNet,loaderVal)
	print ('get ERR and AUC...')
	getEER_AUC()

if __name__ == '__main__':
	trainconfig=configparser.ConfigParser()
	trainconfig.read('./trainconfig.ini')
	
	siameseValFilePath=trainconfig.get('TRAIN','siameseValFilePath')

	main()