import speechpy
import numpy as np
# import scipy.io.wavfile as wav
import soundfile as sf
import random
import math

class Feature_Cube(object):
    """Return a feature cube of desired size.

    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape):
        assert isinstance(cube_shape, (tuple))
        self.cube_shape = cube_shape
        self.num_frames = cube_shape[0]
        self.num_coefficient = cube_shape[1]
        self.num_utterances = cube_shape[2]

    def logenergy_feature(self,wavfile):
        # Get the sound file path
        sound_file_path = wavfile

        ##############################
        ### Reading and processing ###
        ##############################

        
        
        signal, fs = sf.read(sound_file_path)
        


        ###########################
        ### Feature Extraction ####
        ###########################

        # Staching frames
        frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025,
                                                  frame_stride=0.01,
                                                  zero_padding=False)

        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * self.num_coefficient)[:, 1:]

        logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.050, frame_stride=0.01,
                                          num_filters=self.num_coefficient, fft_length=1024, low_frequency=0, high_frequency=None)
        return logenergy

    def getfeature(self, wavfile):
        feature=self.logenergy_feature(wavfile)

        # Feature cube.
        feature_cube = np.zeros((self.num_utterances, self.num_frames, self.num_coefficient), dtype=np.float32)

        # Get some random starting point for creation of the future cube of size (num_frames x num_coefficient x num_utterances)
        # Since we are doing random indexing, the data augmentation is done as well because in each iteration it returns another indexing!
        idx = np.random.randint(feature.shape[0] - self.num_frames, size=self.num_utterances)
        for num, index in enumerate(idx):
            feature_cube[num, :, :] = feature[index:index + self.num_frames, :]

        # return {'feature': feature_cube, 'label': label}
        return feature_cube[None,:,:,:]

class Feature_CubeV2(object):
    """
    feature cube class for extract log energy + 1st logenergy + 2nd logenergy

    """

    def __init__(self,num_coefficient=40):
        self.num_coefficient=num_coefficient

    def pad_zeros(self,sig,frame_sample_length,frame_stride,fs):
        # sig [N,]
        # frame_sample_length,frame_stride in second
        # fs: sample rate
        length_signal=sig.shape[0]
        frame_sample_length=frame_sample_length*fs
        frame_stride=frame_stride*fs
        numframes = (int(math.ceil((length_signal
                                              - frame_sample_length) / frame_stride)))
    #     print(numframes,length_signal,frame_sample_length,frame_stride)

        # Zero padding
        len_sig = int(numframes * frame_stride + frame_sample_length)
        additive_zeros = np.zeros((len_sig - length_signal,))
    #     print (additive_zeros.shape)
        signal = np.concatenate((sig, additive_zeros))
        return signal

    def logenergy_feature(self,wavfile):
        # Get the sound file path
        sound_file_path = wavfile

        ##############################
        ### Reading and processing ###
        ##############################



        signal, fs = sf.read(sound_file_path)
    #     print (fs)
        assert signal.shape[0]>=fs*1.0, 'wavfile too short, wavfile length:{}'.format(signal.shape[0])
        # dynamic start point
        # startpoint=random.randint(0,signal.shape[0]-fs)
        startpoint=0
        signal=signal[startpoint:startpoint+fs] # sample an 1 second serise

        ###########################
        ### Feature Extraction ####
        ###########################
        frame_length=0.025
        frame_stride=0.01
        signal=self.pad_zeros(signal,frame_length,frame_stride,fs)
        logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=frame_length, frame_stride=frame_stride,
                                          num_filters=self.num_coefficient, fft_length=512, low_frequency=0, high_frequency=None)
        return logenergy

    def getfeature(self,wavfile):

        logenergy=self.logenergy_feature(wavfile)

        firstOrder=speechpy.processing.derivative_extraction(logenergy, DeltaWindows=9)
        secondOrder=speechpy.processing.derivative_extraction(firstOrder, DeltaWindows=9)
        
        feat=np.concatenate([logenergy[None,:,:],firstOrder[None,:,:],secondOrder[None,:,:]],axis=0)

        return feat.astype('float32')










