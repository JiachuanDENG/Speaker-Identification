import torch
import torch
import torch.utils.data as Data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  model import models
import argparse
import sys
import configparser
from utils import audioDataset

BATCH_SIZE=32
EPOCH=25

def get_arguments(argv):
	parser = argparse.ArgumentParser(description='Siamese Net for speaker identification')
	# parser.add_argument('file_path', metavar='FILE_PATH',
	#                     help='path of all training files')

	parser.add_argument('-m', '--model', type=int, default=0,
	                    help='0 for classifier, 1 for siamesenet')
	# parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
	#                     help='learning rate for gradient descent (DEFAULT: 1e-4)')
	# parser.add_argument('-i', '--impl', choices=['torch.nn', 'torch.autograd', 'my'], default='my',
	#                     help='choose the network implementation (DEFAULT: my)')
	# parser.add_argument('-g', '--gpu_id', type=int, default=-1,
	#                     help='gpu id to use. -1 means cpu (DEFAULT: -1)')
	# parser.add_argument('-n', '--n_training_examples', type=int, default=-1,
	#                     help='number of training examples used. -1 means all. (DEFAULT: -1)')

	# parser.add_argument('-v', '--verbose', action='store_true', default=False,
	#                     help='show info messages')
	# parser.add_argument('-d', '--debug', action='store_true', default=False,
	#                     help='show debug messages')
	args = parser.parse_args(argv)
	return args


def predict(output):
	values, indices = torch.max(output, 1)
	return indices.data.tolist()
    
    

def perf_measure(y_actual, y_hat):
    """
    evaluate model performance
    """
    TP = 1e-6
    FP = 1e-6
    TN = 1e-6
    FN = 1e-6

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
            TP += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==1:
            TN += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FN += 1
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    F1=2*(precision*recall)/(precision+recall)
    # print ("accuracy:{}, precision:{}, recall:{}, F1:{}, TPR:{}, FPR:{}".format(accuracy,precision,recall,F1,TP/(TP+FN), FP/(FP+TN)))
    return accuracy,recall,F1

def getaccuracy(y_actual,y_hat):
	acc=0.
	for i in range(len(y_hat)):
		if y_actual[i]==y_hat[i]:
			acc+=1
	return acc/len(y_hat)

 
def trainIter4speakerClassify(model,loader,loaderVal,loss_func, optimizer, epochNum=EPOCH, loadModels=False, modelSaveFilePath='./savedEncoder.pkl'):
	if loadModels:
		print ('loading trained model..')
		model.load_state_dict(torch.load(modelSaveFilePath))
	for epoch in range(epochNum):
		print ('*'*10,' EPOCH {} '.format(epoch), '*'*10)
		for step,(x1,y) in enumerate(loader):
			bx1=autograd.Variable(x1)
			by=autograd.Variable(y)
			output=model(bx1)
			loss=loss_func(output,by)
			if step%5==0:
				# print (step)
				preds=predict(output)
				# print (type(preds))
				# accuracy,recall,f1=perf_measure(y.data.tolist(),preds)
				accuracy=getaccuracy(y.data.tolist(),preds)

				# print out validation accuracy every 50 steps
				val_acc_print=''
				yvals,predvals=[],[]
				if step%50==0:
					for xval,yval in loaderVal:
						bxval,byval=autograd.Variable(xval),autograd.Variable(yval)
						outval=model(bxval)
						predval=predict(outval)
						predvals+=predval
						yvals+=yval.data.tolist()
					valaccuracy=getaccuracy(yvals,predvals)
					val_acc_print+=', val accuracy:{}'.format(valaccuracy)

				print ('step {} loss: {}, accuracy:{} {}'.format(step,loss.item(),accuracy,val_acc_print))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		torch.save(model.state_dict(),modelSaveFilePath)


def trainIter4SiameseNet(encoder,siamesenet,loader,loaderVal,\
	loss_func,encoder_optimizer,siames_optimizer,\
	 epochNum=EPOCH,loadModels=False,\
	 encoder_init='./savedEncoder.pkl',saved_encoder='./savedEncoder4Siamese.pkl',\
	 saved_siamese='./savedSiamese.pkl'):
	if loadModels:
		print ('loading saved siamese net...')
		encoder.load_state_dict(torch.load(saved_encoder))
		siamesenet.load_state_dict(torch.load(saved_siamese))
	else:
		print ('init encoder...')
		encoder.load_state_dict(torch.load(encoder_init))

	for epoch in range(epochNum):
		print ('*'*10,' EPOCH {} '.format(epoch), '*'*10)
		for step,(x1,x2,y) in enumerate(loader):
			bx1=autograd.Variable(x1)
			bx2=autograd.Variable(x2)
			by=autograd.Variable(y)
			f1,f2=encoder.getFeature(bx1),encoder.getFeature(bx2)
			output=siamesenet(f1,f2)
			loss=loss_func(output,by)
			if step%5==0:
				# print (step)
				preds=predict(output)
				# accuracy,recall,f1=perf_measure(y.data.tolist(),preds)
				accuracy=getaccuracy(y.data.tolist(),preds)

				val_acc_print=''
				yvals,predvals=[],[]
				if step%50==0:
					for x1val,x2val,yval in loaderVal:
						bx1val,bx2val,byval=autograd.Variable(x1val),autograd.Variable(x2val),autograd.Variable(yval)
						f1val,f2val=encoder.getFeature(bx1val),encoder.getFeature(bx2val)
						outval=siamesenet(f1val,f2val)
						predval=predict(outval)
						predvals+=predval
						yvals+=yval.data.tolist()
					valaccuracy=getaccuracy(yvals,predvals)
					val_acc_print+=', val accuracy:{}'.format(valaccuracy)

				print ('step {} loss: {}, accuracy:{} {}'.format(step,loss.item(),accuracy,val_acc_print))
			encoder_optimizer.zero_grad()
			siames_optimizer.zero_grad()
			loss.backward()
			encoder_optimizer.step()
			siames_optimizer.step()
		torch.save(encoder.state_dict(),saved_encoder)
		torch.save(siamesenet.state_dict(),saved_siamese)


def main4speakerClassify():
	config=configparser.ConfigParser()
	config.read('./model/model_config.ini')
	embedding_size=int(config.get('ENCODER','embedding_size'))
	speaker_num=int(config.get('ENCODER','speaker_num'))

	audioDataSetTr=audioDataset.AudioDataset4speakerClassify(classifierTrainFilePath)
	audioDataSetVal=audioDataset.AudioDataset4speakerClassify(classifierValFilePath)
	loaderTr=Data.DataLoader(
		dataset=audioDataSetTr,
		batch_size=BATCH_SIZE,
		shuffle=True
		)
	loaderVal=Data.DataLoader(
		dataset=audioDataSetVal,
		batch_size=BATCH_SIZE,
		shuffle=True
		)
	encoderNet=models.EncoderNetClassifier(embedding_size=embedding_size,speakersNum=speaker_num)
	optimizer=torch.optim.Adam(encoderNet.parameters(),0.001)
	loss_func=nn.CrossEntropyLoss()
	trainIter4speakerClassify(encoderNet,loaderTr,loaderVal,loss_func,optimizer,loadModels=True)

def main4Siamese():
	
	config=configparser.ConfigParser()
	config.read('./model/model_config.ini')
	embedding_size=int(config.get('ENCODER','embedding_size'))
	speaker_num=int(config.get('ENCODER','speaker_num'))
	try:
		audioDataSet=audioDataset.AudioDataset4Siamese(siameseFilePath)
		audioDataSetVal=audioDataset.AudioDataset4Siamese(siameseValFilePath)
		loader=Data.DataLoader(
		dataset=audioDataSet,
		batch_size=BATCH_SIZE,
		shuffle=True
		)
		loaderVal=Data.DataLoader(
		dataset=audioDataSetVal,
		batch_size=BATCH_SIZE,
		shuffle=True
		)
	except:
		raise NotImplementedError

	encoderNet=models.EncoderNetClassifier(embedding_size=embedding_size,speakersNum=speaker_num)
	siameseNet=models.SiameseNet(encoder_embeddingsize=embedding_size)
	encoder_optimizer=torch.optim.Adam(encoderNet.parameters(),0.001)
	siames_optimizer=torch.optim.Adam(siameseNet.parameters(),0.001)
	loss_func=nn.CrossEntropyLoss()
	trainIter4SiameseNet(encoderNet,siameseNet,loader,loaderVal,\
	loss_func,encoder_optimizer,siames_optimizer,\
	 loadModels=False)




if __name__ == '__main__':
	args=get_arguments(sys.argv[1:])
	trainconfig=configparser.ConfigParser()
	trainconfig.read('./trainconfig.ini')
	classifierTrainFilePath=trainconfig.get('TRAIN','classifierTrainFilePath')
	classifierValFilePath=trainconfig.get('TRAIN','classifierValFilePath')
	siameseFilePath=trainconfig.get('TRAIN','siameseFilePath')
	siameseValFilePath=trainconfig.get('TRAIN','siameseValFilePath')
	if args.model==0:
		main4speakerClassify()
	elif args.model==1:
		main4Siamese()
	else:
		raise NotImplementedError


