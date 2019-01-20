import speechpy
import numpy as np
# import scipy.io.wavfile as wav
import soundfile as sf


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