# MACR-Afford: Weakly Supervised Multimodal Affordance Grounding via Multi-Branch Attention Enhancement and CoT Multi-Stage Reasoning


## Usage

### 1. Requirements

Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.6

```
pip install -r requirements.txt
```

### 2. Dataset

Download the AGD20K dataset
from [ [Google Drive](https://drive.google.com/file/d/1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk/view?usp=sharing)].

### 3. Train and Test

#### 3.1 Train
The training code is being organized.
#### 3.2 Test
Our pretrained model can be downloaded
  from [Google Drive](https://drive.google.com/file/d/1AsOwHFf_31O_tKJVPOMn5vj6DSzT0fEi/view?usp=sharing). Run following commands to start training or testing:

```
python train.py --data_root <PATH_TO_DATA>  --model_file <pretrained model path> --divide <Seen / Unseen> --gpu <gpu_id>
```



## Anckowledgement

This repo is based on [Locate](https://github.com/Reagan1311/LOCATE)
. Thanks for their great work!