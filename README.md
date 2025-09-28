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


## Experiment Results

After executing the training script, a new directory with the format `exp_*/` will be created in the `runs/` directory. This directory contains the experiment logs and information related to the training run.

- The experiment results will be stored in the `results/` directory. 

- The trained models will be saved in the `models/` directory.

- The training loss and other metrics are recorded in the `logs/` directory. You can use TensorBoard to visualize and analyze the training progress by running the following command in the project directory:
```
tensorboard --logdir=logs/
```

## Scripts (Before Training)
1. `preprocess`: extract frames, split train/test
  ```bash
  python scripts/preprocess.py --input_dir ./dataset/how2sign-zhewen --output_dir ./dataset/customized_dataset
  ```
2. `pkl2json`: transform pkl to json (avoid version conflict of numpy)
3. `img_stats.py`: calculate the mean & std for datasets (used by normalization later)
4. training script
  ```bash
  python train.py --ds_name "customized" --which_g "7" --ema_rate 0.999 --input_type "heatmaps" --hand_l1 True
  ```

## Kps Index Choice
* My choice, Refer this code segment:
```py
# 按照“手部 > 面部 > 上半身”的优先级进行选择
upper_body_indices = [0, 5, 6, 7, 8]  # 鼻子, 左右肩, 左右肘
face_expression_indices = (
    list(range(27, 37)) +  # 眉毛 (10)
    list(range(37, 49)) +  # 眼睛 (12)
    list(range(49, 56)) +  # 鼻子 (7)
    list(range(56, 68)) +  # 外唇 (12)
    list(range(17, 25))    # 部分脸颊轮廓 (8)
)
hand_indices = list(range(91, 133))  # 左右手 (42)

# 组合并排序，最终得到一个有序的索引列表
self.selected_indices = torch.tensor(sorted(
    upper_body_indices + face_expression_indices + hand_indices
))
```

* COCO-Wholebody-133-joints Index Def:
> 以下是 COCO-WholeBody 133 joints 的详细 index 对应部位划分（从 0 开始计数）：
>
>  ⸻
>
>  🧍‍♂️ Body（人体主干, 17 joints）
>
>  0–16
>
>  范围 部位
>  0 鼻子
>  1–4 眼睛和耳朵（左眼、右眼、左耳、右耳）
>  5–10 肩膀、肘、腕（左肩、右肩、左肘、右肘、左腕、右腕）
>  11–16 髋、膝、踝（左髋、右髋、左膝、右膝、左踝、右踝）
>
>
>  ⸻
>
>  ✋ Hands（手部, 左右各 21 joints）
>
>  左手：91–111  
>  右手：112–132
>
>  每只手的 21 个关键点依次为：
>  • 0：手腕
>  • 1–4：拇指（四个关节）
>  • 5–8：食指
>  • 9–12：中指
>  • 13–16：无名指
>  • 17–20：小指
>
>  ⸻
>
>  😊 Face（面部, 68 joints）
>
>  17–84
>
>  区域 索引范围 说明
>  脸轮廓 17–26 下颌、脸颊
>  眉毛 27–36 左右眉毛
>  眼睛 37–48 左右眼
>  鼻子 49–55 鼻梁与鼻尖
>  嘴唇 56–84 外唇与内唇轮廓
>
>
>  ⸻
>
>  🦶 Feet（脚部, 6 joints）
>
>  85–90
>
>  左右脚的脚趾与脚掌关键点。
>
>  ⸻
>
>  ✅ 推荐用于手语识别的索引子集
>
>  区域 索引范围 说明
>  上半身（身体+头部） 0–10 鼻、眼、耳、肩、肘、腕
>  手部（最关键） 91–132 左右手各 21 点
>  面部（可选） 17–26 + 49–55 + 56–63 眉毛、鼻子、嘴部表情
>
>  这样总共约 60–80 个点，足以覆盖手语识别的主要信息。
>
>  ⸻