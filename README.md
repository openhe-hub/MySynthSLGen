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

- `Phoenix` dataset: <br>
The dataset consists of multiple tar files: `train.tar`, `base_*.tar`, and `test.tar`


- `Bosphorus` dataset: <br>
The dataset contains multiple tar files for training and testing, as well as a `base_*.tar` files. `base_*.tar` needs to be decompressed and placed under our `/netscratch/$USER` directory by ourselves.

<br>
Please check the codes under `DS_loader` to ensure that the paths are properly configured.



## Training the Model

- To train the model, you can use the following commands:
```
python train.py --ds_name "<dataset_name>" --which_g "<generator_type>" --ema_rate <ema_rate_value> --input_type "<input_type>"
```
> ### Parameters

- **`--ds_name`**: Specifies the dataset for training. Available options are:
  - `SynthSL`
  - `Phoenix`
  - `Bosphorus`

- **`--which_g`**: Determines the generator architecture to use. Choose from:
  - `0` through `9`

- **`--ema_rate`**: Sets the exponential moving average rate for model weights. Use a value such as:
  - `0.9999`

- **`--input_type`**: Defines the type of input data. Available options are:
  - `heatmaps`
  - `depth`
  - `segm`
  - `normal`



>### Example Usage:
```
python train.py --ds_name "SynthSL" --which_g "7" --ema_rate 0.999 --input_type "heatmaps"
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
