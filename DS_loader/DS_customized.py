import os
import pickle
import joblib
import glob
import numpy as np
import json
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.datapipes import functional_datapipe

@functional_datapipe("read_video_and_kps")
class VideoDataProcessor(IterDataPipe):
    """
    A functional DataPipe that reads frames from a directory, pairs them with
    keypoints from a .pkl file, and yields processed tensors in the expected format.
    """
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe

        self.img_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ============== 新增代码: 定义专用于手语识别的96个关键点索引 ==============
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
        # 确认我们不多不少正好选择了96个点
        assert len(self.selected_indices) == 96, "The number of selected keypoints must be 96."
        # ============================================================================


    def __iter__(self):
        """
        Iterates over (frames_dir, pkl_path) pairs from the source datapipe.
        """
        for frames_dir, pkl_path in self.source_datapipe:
            
            try:
                json_path = pkl_path.replace('pkl', 'json')
                with open(json_path, 'r') as f:
                    # 加载所有关键点并立即转换为张量
                    all_keypoints = torch.tensor(json.load(f)['kps'], dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Could not load keypoint file {json_path}. Error: {e}. Skipping sequence.")
                continue

            try:
                selected_keypoints = torch.index_select(all_keypoints, 2, self.selected_indices)
            except IndexError as e:
                print(f"Warning: Indexing error for {pkl_path}. Check keypoint dimensions. Skipping sequence. Error: {e}")
                continue
            
            frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.*')))
            
            if not frame_files:
                continue

            # Sanity check
            if len(frame_files) != selected_keypoints.shape[0]:
                print(f"Warning: Mismatch in sequence {os.path.basename(frames_dir)}. Skipping sequence.")
                continue

            # a. 将视频的第一帧作为 "base" 帧
            base_img_path = frame_files[0]
            base_img = self.transform(Image.open(base_img_path).convert('RGB'))
            base_kps = selected_keypoints[0].squeeze(0) * self.img_size[0] # [96, 2]
            base_heatmaps = self.kps_to_heatmaps(base_kps)

            # b. 为 depth, segm, normal 创建全零的占位符 (dummy) 张量
            dummy_tensor = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.float32)
            
            # c. Iterate through each frame (now considered the "target" frame)
            for i, frame_path in enumerate(frame_files):
                try:
                    img_tensor = self.transform(Image.open(frame_path).convert('RGB')) # [3, 256, 256]

                    kps = selected_keypoints[i].squeeze(0) * self.img_size[0] # [96, 2]
                    
                    heatmaps = self.kps_to_heatmaps(kps) # [96, 256, 256]
                    
                    yield (
                        base_img, base_heatmaps, dummy_tensor, dummy_tensor, dummy_tensor,
                        img_tensor, kps, heatmaps, dummy_tensor, dummy_tensor, dummy_tensor
                    )
                except Exception as e:
                    print(f"Warning: Error processing frame {frame_path}. Error: {e}. Skipping frame.")
                    continue
    
    def kps_to_heatmaps(self, keypoints: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
        heatmaps = []
        for joint in keypoints:
            mu_x = int(joint[0])
            mu_y = int(joint[1])
            if 0 <= mu_x < self.img_size[0] and 0 <= mu_y < self.img_size[1]:
                x = torch.arange(0, self.img_size[0], dtype=torch.float32)
                y = torch.arange(0, self.img_size[1], dtype=torch.float32).unsqueeze(-1)
                heatmap = torch.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
            else:
                heatmap = torch.zeros(self.img_size)
            heatmaps.append(heatmap)
        return torch.stack(heatmaps)


class CustomDataset:
    def __init__(self, root_dir: str, train: bool = True):
        """
        Initializes the dataset by finding all valid samples.

        Args:
            root_dir (str): The root directory of the structured dataset,
                            which contains 'train' and 'test' subdirectories.
            train (bool): If True, load the training set, otherwise load the test set.
        """
        self.root_dir = root_dir
        self.is_train = train
        self.data_dir = os.path.join(self.root_dir, 'train' if self.is_train else 'test')

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Scan the filesystem to create a list of (frame_folder, pkl_file) pairs
        self.samples = self._create_sample_list()
        print(f"Found {len(self.samples)} samples in '{self.data_dir}'.")

    def _create_sample_list(self) -> list:
        """
        Scans the data directory to find pairs of (frame_directory, pkl_file).
        """
        samples = []
        # Find all .pkl files in the target directory (train or test)
        pkl_files = glob.glob(os.path.join(self.data_dir, '*.pkl'))

        for pkl_path in pkl_files:
            # Infer the corresponding frame directory name from the pkl filename
            base_name = os.path.splitext(os.path.basename(pkl_path))[0]
            frames_dir = os.path.join(self.data_dir, base_name)

            # Only add the pair if the corresponding frame directory actually exists
            if os.path.isdir(frames_dir):
                samples.append((frames_dir, pkl_path))
        
        return samples

    def create_datapipe(self) -> IterDataPipe:
        """
        Builds and returns a torchdata DataPipe for this dataset.
        """
        # 1. Start the pipeline with our list of samples
        pipe = IterableWrapper(self.samples)

        # 2. If in training mode, shuffle the order of the videos
        if self.is_train:
            pipe = pipe.shuffle()
        
        # 3. For distributed training, ensure each process gets a unique shard of data
        pipe = pipe.sharding_filter()
        
        # 4. Apply our custom data processing function to read and transform the data
        pipe = pipe.read_video_and_kps()
        
        return pipe

# example
if __name__ == '__main__':
    
    # IMPORTANT: Change this path to the root of your NEWLY CREATED structured dataset
    DATASET_ROOT_DIR = "dataset/customized_dataset" 

    # Check if the dataset directory exists before proceeding
    if not os.path.isdir(DATASET_ROOT_DIR) or not os.path.isdir(os.path.join(DATASET_ROOT_DIR, 'train')):
        print("="*50)
        print(f"ERROR: Dataset not found at '{DATASET_ROOT_DIR}'")
        print("Please update the 'DATASET_ROOT_DIR' variable in this script")
        print("to point to the folder created by 'prepare_dataset.py'.")
        print("="*50)
    else:
        # 1. Instantiate the dataset for the training set
        print(f"Loading dataset from: {DATASET_ROOT_DIR}")
        train_dataset = CustomDataset(root_dir=DATASET_ROOT_DIR, train=True)

        # 2. Create the data pipeline
        train_datapipe = train_dataset.create_datapipe()

        # 3. Wrap the DataPipe with PyTorch's DataLoader for batching and multi-threading
        #    Note: batch_size refers to the number of frames per batch, not videos.
        train_loader = DataLoader(
            dataset=train_datapipe,
            batch_size=32,
            num_workers=4  # Set to 0 on Windows if you encounter issues
        )

        # 4. Iterate through the DataLoader to get batches of data
        for batch_idx, (base_img, base_heatmap, base_depth, base_segm, base_normal, img, kps, heatmap, depth, segm, normal) in enumerate(train_loader):
            print(f"--- Batch {batch_idx + 1} ---")
            print(f"  Images tensor shape:    {img.shape}")    # e.g., torch.Size([32, 3, 256, 256])
            print(f"  Keypoints tensor shape: {kps.shape}")  # e.g., torch.Size([32, num_kps, 2])
            print(f"  Heatmaps tensor shape:  {heatmap.shape}")   # e.g., torch.Size([32, num_kps, 256, 256])

            # Stop after showing a few batches for demonstration
            if batch_idx >= 2:
                break
                
        print("\nDemonstration finished successfully.")