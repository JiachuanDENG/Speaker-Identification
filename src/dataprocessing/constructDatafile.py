import os
import sys
import random
import scipy.io.wavfile as wav
from tqdm import tqdm

def samplefiles4speakerclassify(audiodir,f,fval,speakerIds,sampleNum):
	# speakerIds to int ids

	speakerIds2Idx={}
	idx=0
	for speakerid in speakerIds:
		if speakerid not in speakerIds2Idx:
			speakerIds2Idx[speakerid]=idx
			idx+=1

	alreadyPickedfiles=set()
	trainSampleNum=int(sampleNum*0.8)
	valSampleNum=sampleNum - trainSampleNum

	i=0
	
	while i < trainSampleNum:
		# sample a speaker
		speakerid=random.choice(speakerIds)
		speakeridx=speakerIds2Idx[speakerid]
		speakerpath=os.path.join(audiodir,speakerid)
		# sample two wavfile from same speaker
		wav1=random.choice([os.path.join(speakerpath,wavfile ) for wavfile in os.listdir(speakerpath) if '.wav' in wavfile ])
		if wav1 in alreadyPickedfiles:
			continue
		try:
			with open(wav1, 'rb') as f1:
				riff_size, _ = wav._read_riff_chunk(f1)
				file_size = os.path.getsize(wav1)
				# Assertion error. 
			assert riff_size == file_size and os.path.getsize(wav1) > 25000, "Bad file!"

		except :
			# print (wav1)
			print('file %s is corrupted or too short!' % wav1)
			continue
		alreadyPickedfiles.add(wav1)
		f.write('{} {}\n'.format(wav1,speakeridx))
		i+=1

	alreadyPickedVal=set()
	i=0
	while i<valSampleNum:
		# sample a speaker
		speakerid=random.choice(speakerIds)
		speakeridx=speakerIds2Idx[speakerid]
		speakerpath=os.path.join(audiodir,speakerid)
		# sample two wavfile from same speaker
		wav1=random.choice([os.path.join(speakerpath,wavfile ) for wavfile in os.listdir(speakerpath) if '.wav' in wavfile ])
		if wav1 in alreadyPickedVal:
			continue
		if wav1 in alreadyPickedfiles:
			continue
		try:
			with open(wav1, 'rb') as f1:
				riff_size, _ = wav._read_riff_chunk(f1)
				file_size = os.path.getsize(wav1)
				# Assertion error. 
			assert riff_size == file_size and os.path.getsize(wav1) > 25000, "Bad file!"

		except :
			# print (wav1)
			print('file %s is corrupted or too short!' % wav1)
			continue
		alreadyPickedVal.add(wav1)
		fval.write('{} {}\n'.format(wav1,speakeridx))
		i+=1

	print ('number of speakers: {}'.format(len(speakerIds)))

	f.close()
	fval.close()





def samplefiles4Siamese(audiodir,f,speakerIds,sampleNum):
	genuineSamplesNum=sampleNum//2
	imposterSamplesNum=sampleNum -  genuineSamplesNum
	print ('generate genuine pairs...')
	i=0
	while i < genuineSamplesNum:
		# sample a speaker
		speakerid=random.choice(speakerIds)
		speakerpath=os.path.join(audiodir,speakerid)
		# sample two wavfile from same speaker
		wav1,wav2=random.sample([os.path.join(speakerpath,wavfile ) for wavfile in os.listdir(speakerpath) if '.wav' in wavfile ],2)
		try:
			with open(wav1, 'rb') as f1:
				riff_size, _ = wav._read_riff_chunk(f1)
				file_size = os.path.getsize(wav1)

				# Assertion error.
			assert riff_size == file_size and os.path.getsize(wav1) > 25000, "Bad file!"

		except:
			print('file %s is corrupted or too short!' % wav1)
			continue

		try:
			with open(wav2, 'rb') as f2:
				riff_size, _ = wav._read_riff_chunk(f2)
				file_size = os.path.getsize(wav2)

			# Assertion error.
			assert riff_size == file_size and os.path.getsize(wav2) > 25000, "Bad file!"

		except:
			print('file %s is corrupted or too short!' % wav2)
			continue

		f.write('{} {} 1\n'.format(wav1,wav2))
		i+=1


	print ('generate imposter pairs...')
	j=0
	while j<imposterSamplesNum:
		# sample a speaker
		onespeakerid=random.choice(speakerIds)
		# sample another speaker 
		while True:
			antspeakerid=random.choice(speakerIds)
			if antspeakerid!=onespeakerid:
				break
		onespeakerPath=os.path.join(audiodir,onespeakerid)
		antspeakerPath=os.path.join(audiodir,antspeakerid)
		wav1=random.choice([os.path.join(onespeakerPath,wavfile)for wavfile in os.listdir(onespeakerPath) if '.wav' in wavfile])
		wav2=random.choice([os.path.join(antspeakerPath,wavfile) for wavfile in os.listdir(antspeakerPath) if '.wav' in wavfile])
		try:
			with open(wav1, 'rb') as f1:
				riff_size, _ = wav._read_riff_chunk(f1)
				file_size = os.path.getsize(wav1)

			# Assertion error.
			assert riff_size == file_size and os.path.getsize(wav1) > 25000, "Bad file!"

		except:
			print('file %s is corrupted or too short!' % wav1)
			continue

  
		try:
			with open(wav2, 'rb') as f2:
				riff_size, _ = wav._read_riff_chunk(f2)
				file_size = os.path.getsize(wav2)

			# Assertion error.
			assert riff_size == file_size and os.path.getsize(wav2) > 25000, "Bad file!"

		except:
			print('file %s is corrupted or too short!' % wav2)
			continue

		f.write('{} {} 0\n'.format(wav1,wav2))
		j+=1

	f.close()


def main(args):
	if len(args)!= 3:
		print('usage: python3 constructDatafile.py <audio dir> <train speaker ratio> <training sample number>')
		# audio dir
		# 	| id 1
		# 		| 0.wav
		# 		| 1.wav
		# 	| id2
		# 		| 0.wav
		sys.exit(1)
	audio_dir=args[0]
	trainSpeakerRatio=float(args[1])
	trainSampleNum=int(args[2])
	valSampleNum=int(trainSampleNum*(1 - trainSpeakerRatio)/trainSpeakerRatio)

	allspeakerids=[id for id in os.listdir(audio_dir) if ".DS_Store" not in id]

	trainSpeakerIds=random.sample(allspeakerids,int(len(allspeakerids)*trainSpeakerRatio))

	valSpeakerIds=[id for id in allspeakerids if id not in trainSpeakerIds]

	f_trainfiles=open('train_file_path.txt','w')
	fsiamese_trainfiles=open('train_siamese_file_path.txt','w')
	f_valfiles=open('val_file_path.txt','w')
	fsiamese_Evalfiles=open('eval_siamese_file_path.txt','w')
	print ('training files...')
	# samplefiles(audio_dir,f_trainfiles,trainSpeakerIds,trainSampleNum)
	samplefiles4speakerclassify(audio_dir,f_trainfiles,f_valfiles,trainSpeakerIds,trainSampleNum)

	samplefiles4Siamese(audio_dir,fsiamese_trainfiles,trainSpeakerIds,trainSampleNum)

	print ('eval files...')

	samplefiles4Siamese(audio_dir,fsiamese_Evalfiles,valSpeakerIds,valSampleNum)
		




if __name__ == '__main__':
	main(sys.argv[1:])


