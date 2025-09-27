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

# ========================================================================
#   Module 1: Data Processor (Functional DataPipe)
#   Reads a (frame folder, pkl file) pair and yields processed frames.
# ========================================================================

@functional_datapipe("read_video_and_kps")
class VideoDataProcessor(IterDataPipe):
    """
    A functional DataPipe that reads frames from a directory, pairs them with
    keypoints from a .pkl file, and yields processed tensors in the expected format.
    """
    def __init__(self, source_datapipe: IterDataPipe) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe

        # Define the image preprocessing pipeline
        self.img_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        """
        Iterates over (frames_dir, pkl_path) pairs from the source datapipe.
        """
        for frames_dir, pkl_path in self.source_datapipe:
            
            # 1. Load all keypoints for the entire video sequence at once
            try:
                json_path = pkl_path.replace('pkl', 'json')
                with open(json_path, 'r') as f:
                    all_keypoints = json.load(f)['kps']
            except Exception as e:
                print(f"Warning: Could not load pickle file {pkl_path}. Error: {e}. Skipping sequence.")
                continue

            # 2. Get all frame image paths and sort them to ensure correct order
            frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.*')))
            
            if not frame_files:
                continue

            # 3. Sanity check
            if len(frame_files) != len(all_keypoints):
                print(f"Warning: Mismatch in sequence {os.path.basename(frames_dir)}. Skipping sequence.")
                continue

            # ============== 新增代码: 创建 "Base" 帧和 "Dummy" 张量 ==============
            
            # a. 将视频的第一帧作为 "base" 帧 (源)
            base_img_path = frame_files[0]
            base_img = self.transform(Image.open(base_img_path).convert('RGB'))
            base_kps = torch.tensor(all_keypoints[0], dtype=torch.float32) * self.img_size[0]
            base_kps = base_kps[0][:96]
            base_heatmaps = self.kps_to_heatmaps(base_kps)

            # b. 为 depth, segm, normal 创建全零的占位符 (dummy) 张量
            #    形状为 [1, H, W]，模仿单通道图像
            dummy_tensor = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.float32)
            
            # =================================================================

            # 4. Iterate through each frame (now considered the "target" frame)
            for i, frame_path in enumerate(frame_files):
                try:
                    # Open, convert, and transform the target image
                    img_tensor = self.transform(Image.open(frame_path).convert('RGB'))  # (3, 256, 256)

                    # Get the corresponding keypoints and scale them
                    kps = torch.tensor(all_keypoints[i], dtype=torch.float32) * self.img_size[0]
                    kps = kps[0][:96]  # (133, 2)
                    
                    # Convert keypoint coordinates into heatmaps
                    heatmaps = self.kps_to_heatmaps(kps)  # (133, 256, 256)
                    
                    # ============== 修改产出 (yield) 的数据结构 ==============
                    # 按照 GANPipe 期望的11个元素的顺序来产出
                    yield (
                        base_img,         # base_img
                        base_heatmaps,    # base_heatmap
                        dummy_tensor,     # base_depth (dummy)
                        dummy_tensor,     # base_segm (dummy)
                        dummy_tensor,     # base_normal (dummy)
                        img_tensor,       # img
                        kps,              # coord (就是我们的 kps)
                        heatmaps,         # heatmap
                        dummy_tensor,     # depth (dummy)
                        dummy_tensor,     # segm (dummy)
                        dummy_tensor      # normal (dummy)
                    )
                    # ========================================================

                except Exception as e:
                    print(f"Warning: Error processing frame {frame_path}. Error: {e}. Skipping frame.")
                    continue
    
    def kps_to_heatmaps(self, keypoints: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
        """
        Converts keypoint coordinates into a stack of Gaussian heatmaps.
        """
        heatmaps = []
        for joint in keypoints:
            mu_x = int(joint[0])
            mu_y = int(joint[1])
            
            # Only generate a heatmap if the keypoint is within the image bounds
            if 0 <= mu_x < self.img_size[0] and 0 <= mu_y < self.img_size[1]:
                # Create coordinate grids
                x = torch.arange(0, self.img_size[0], dtype=torch.float32)
                y = torch.arange(0, self.img_size[1], dtype=torch.float32).unsqueeze(-1)
                
                # Calculate the Gaussian distribution
                heatmap = torch.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
            else:
                # If keypoint is out of bounds, return a zero heatmap
                heatmap = torch.zeros(self.img_size)
            
            heatmaps.append(heatmap)

        return torch.stack(heatmaps)


# ========================================================================
#   Module 2: Dataset Definition and DataPipe Creation
#   Scans the filesystem for (frame_folder, pkl_file) pairs and
#   builds the torchdata pipeline.
# ========================================================================

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

# ========================================================================
#   Module 3: Usage Example
#   Demonstrates how to use the CustomDataset class and DataLoader.
# ========================================================================

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
        for batch_idx, (images, keypoints, heatmaps) in enumerate(train_loader):
            print(f"--- Batch {batch_idx + 1} ---")
            print(f"  Images tensor shape:    {images.shape}")    # e.g., torch.Size([32, 3, 256, 256])
            print(f"  Keypoints tensor shape: {keypoints.shape}")  # e.g., torch.Size([32, num_kps, 2])
            print(f"  Heatmaps tensor shape:  {heatmaps.shape}")   # e.g., torch.Size([32, num_kps, 256, 256])

            # Stop after showing a few batches for demonstration
            if batch_idx >= 2:
                break
                
        print("\nDemonstration finished successfully.")