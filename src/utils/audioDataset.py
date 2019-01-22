# construct a dataset so that we can use it in pytorch DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
from .extractFeature import  Feature_Cube

class AudioDataset4Siamese():
    """Audio dataset."""

    def __init__(self, files_path,cube_shape=(40,40,10)):
        """
        Args:
            files_path (string): Path to the .txt file which the address of files are saved in it.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        # Open the .txt file and create a list from each line.
        with open(files_path, 'r') as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        list_files = []
        labels=[]
        for x in content:
            contentInfos= x.strip().split()
            sound_file_path1, sound_file_path2,label = contentInfos[0],contentInfos[1],int(contentInfos[2])
            list_files.append((sound_file_path1,sound_file_path2))
            labels.append(label)

        # Save the correct and healthy sound files to a list.
        self.sound_files = list_files
        self.labels=labels

        self.featureCube=Feature_Cube(cube_shape)

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        # Get the sound file path
        sound_file_path1,sound_file_path2 = self.sound_files[idx][0],self.sound_files[idx][1]
        label=self.labels[idx]

        ##############################
        ### Reading and processing ###
        ##############################
        feat1,feat2=self.featureCube.getfeature(sound_file_path1),self.featureCube.getfeature(sound_file_path2)


        return torch.from_numpy(feat1),torch.from_numpy(feat2),label


class AudioDataset4speakerClassify():
    """Audio dataset."""

    def __init__(self, files_path,cube_shape=(40,40,10)):
        """
        Args:
            files_path (string): Path to the .txt file which the address of files are saved in it.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        # Open the .txt file and create a list from each line.
        with open(files_path, 'r') as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        list_files = []
        labels=[]
        for x in content:
            contentInfos= x.strip().split()
            sound_file_path1,label = contentInfos[0],int(contentInfos[1])
            list_files.append(sound_file_path1)
            labels.append(label)

        # Save the correct and healthy sound files to a list.
        self.sound_files = list_files
        self.labels=labels

        self.featureCube=Feature_Cube(cube_shape)

    def __len__(self):
        return len(self.sound_files)

    def __getitem__(self, idx):
        # Get the sound file path
        sound_file_path1 = self.sound_files[idx]
        label=self.labels[idx]

        ##############################
        ### Reading and processing ###
        ##############################
        feat1=self.featureCube.getfeature(sound_file_path1)
        return torch.from_numpy(feat1),label




