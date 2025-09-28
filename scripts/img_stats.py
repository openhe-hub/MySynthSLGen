import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
from PIL import Image
from tqdm import tqdm

# ========================================================================
#   一个简化的数据集类，仅用于读取图片
# ========================================================================
class ImageDatasetForStats(Dataset):
    """
    一个简化的数据集，用于遍历训练集中的所有帧图像。
    """
    def __init__(self, root_dir: str):
        # 我们只关心训练集 (train) 的统计数据
        self.data_dir = os.path.join(root_dir, 'train')
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Training data directory not found: {self.data_dir}\n"
                "Please ensure your dataset is structured correctly and the root_dir is correct."
            )
        
        # 找到所有子文件夹中的所有图片文件（例如 .jpg, .png）
        self.image_files = glob.glob(os.path.join(self.data_dir, '*', '*.*'))
        
        # 定义一个简单的 transform，只将图片转为 [0, 1] 范围的 Tensor
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

# ========================================================================
#   主计算逻辑
# ========================================================================
def calculate_mean_std(dataset_root_dir: str, batch_size: int = 64, num_workers: int = 4):
    """
    计算并打印数据集的均值和标准差。

    Args:
        dataset_root_dir (str): 数据集的根目录。
        batch_size (int): 用于计算的批大小。
        num_workers (int): 数据加载器的工作线程数。
    """
    print(f"Calculating stats for dataset at: {dataset_root_dir}")
    
    # 1. 实例化数据集和加载器
    dataset = ImageDatasetForStats(root_dir=dataset_root_dir)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    if len(dataset) == 0:
        print("Error: No images found in the training directory. Cannot calculate stats.")
        return

    # 2. 初始化变量
    # 用于累加每个通道的像素值总和
    channel_sum = torch.zeros(3)
    # 用于累加每个通道的像素值平方和
    channel_sum_sq = torch.zeros(3)
    # 像素总数
    num_pixels = 0

    # 3. 遍历数据集
    for images in tqdm(loader, desc="Calculating Mean/Std"):
        # images 的形状: [B, C, H, W]
        # B: batch_size, C: 3 (RGB), H: height, W: width
        
        # 计算批次中的像素总数并累加
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        
        # 沿着 batch, height, width 维度求和，保留 channel 维度
        # 结果形状为 [3]
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images ** 2, dim=[0, 2, 3])

    # 4. 计算最终的均值和标准差
    mean = channel_sum / num_pixels
    # Var(X) = E[X^2] - (E[X])^2
    var = (channel_sum_sq / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    # 5. 打印结果
    print("\n" + "="*50)
    print("Calculation Complete!")
    print(f"  - Total images processed: {len(dataset)}")
    print(f"  - Total pixels: {num_pixels}")
    print("\nCopy these values into your VideoDataProcessor class:")
    print(f"  mean={mean.tolist()}")
    print(f"  std={std.tolist()}")
    print("="*50)


if __name__ == '__main__':
    DATASET_ROOT_DIR = "dataset/customized_dataset"
    
    if not os.path.isdir(DATASET_ROOT_DIR) or not os.path.isdir(os.path.join(DATASET_ROOT_DIR, 'train')):
        print(f"ERROR: Dataset not found at '{DATASET_ROOT_DIR}'")
        print("Please update the 'DATASET_ROOT_DIR' variable in this script.")
    else:
        calculate_mean_std(DATASET_ROOT_DIR)