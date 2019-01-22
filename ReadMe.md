***Author***: Jiachuan Deng

### Goal
On-going project experimental codes for building a real time one-shot deep learning speaker identification system.

### File Structure
``` bash
Speaker-Identification
|-- ReadMe.md
|-- data
|   |-- voxData (raw data downloaded from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
|   |-- voxVadData (raw data after vad)
|   
`-- src
    |-- dataprocessing
    |    |-- config.ini (config file)
    |    |-- run_dataprocessing.py (exe code)
    |    |
    |    `-- vad.py (vad toolkit)
    |
    |-- model
    |     |
    |     `-- models.py (definemodels)
    |-- utils
    |     |-- autioDataset.py (Data set defined in pytorch dataset style)
    |     |
    |     `-- extractFeature.py (extract MFCC feature from raw audio)
    |
    `-- trainModel.py (executable script to train model)
  
```
### Run code
#### 0. data processing
under dataprocessing directory, do:
``` bash
python3 run_dataprocessing.py
```
Make sure the paths in  config.ini are consistent with your local files.

The script will result in several audio chunks saved under ```/data/voxVadData```, and a ```train_file_path.txt```,```val_file_path.txt``` stored under ```/dataprocessing```.

#### 1. train model
under ```Speaker-Identification``` directory, do:
i. for train classifier encoder model:
``` bash
python3 trainModel.py -m 0 
```

ii. for train siamesenet model:
``` bash
python3 trainModel.py -m 1
```

