# Sign Language Production

## Project Overview

Sign Language Production (SLP) has wide-ranging uses in education, translation, etc. Despite its many uses, SLP faces a number of challenges that might impact its results. For example, the accuracy of converting sentences to pose representations, or how well we can synthesize images from the pose representations. We found that Sign language images often suffer from blurriness in hands and faces. Therefore, we dedicated our efforts to the second challenge, synthesis of images. <br>

In this task, the model takes a reference image of the signer and the desired pose as input, and then generates a target image in which the signer is performing the specified pose.


## Installation

Here are the steps to install and set up the project: <br>

1. Clone the GitHub repository of this project to your local environment: <br>
```
git clone git@github.com:CYWangKL/SLP.git
```

2. Create and activate a virtual environment (optional): <br>
```
conda create --name slp
conda activate slp
```

3. Install requirements
```
cd SLP
pip install -r requirements.txt
```

## Datasets

- `SynthSL` dataset: <br>
The dataset consists of multiple tar files: `train.tar`, `base_*.tar`, and `test.tar`

- `Bosphorus` dataset: <br>
The dataset contains multiple tar files for training and testing, as well as a `base_*.tar` files. `base_*.tar` needs to be decompressed and placed under our `/netscratch/$USER` directory by ourselves.

<br>
Please check the codes under `DS_loader` to ensure that the paths are properly configured.



## Training the Model

- To train the model, you can use the following commands:
```
python train.py
```

For more details, please refer to the thesis.


## Experiment Results

After executing the training script, a new directory with the format `exp_*/` will be created in the `runs/` directory. This directory contains the experiment logs and information related to the training run.

- The experiment results will be stored in the `results/` directory. 

- The trained models will be saved in the `models/` directory.

- The training loss and other metrics are recorded in the `logs/` directory. You can use TensorBoard to visualize and analyze the training progress by running the following command in the project directory:
```
tensorboard --logdir=logs/
```
