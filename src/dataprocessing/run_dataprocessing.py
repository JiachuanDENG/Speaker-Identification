import os
import sys
import configparser




def main():
	config=configparser.ConfigParser()
	config.read('./config.ini')
	vad_aggressiveness=config.get('VAD','aggressiveness')
	vad_sr=config.get('VAD','sample_rate')
	audio_dir=config.get('VAD','audio_dir')
	train_speaker_ratio=config.get('ConstructFile','train_speaker_ratio')
	train_sample_num=config.get('ConstructFile','train_sample_num')

	if os.path.exists(audio_dir):
		os.system('rm -r {}'.format(audio_dir))
	
	os.system('python3 vad.py {} {} {}'.format(vad_aggressiveness,vad_sr,audio_dir))
	os.system('python3 constructDatafile.py {} {} {}'.format(audio_dir,train_speaker_ratio,train_sample_num))

if __name__ == '__main__':

	main()