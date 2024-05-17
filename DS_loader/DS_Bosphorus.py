from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
import io
import pickle
import tarfile
import numpy as np
import cv2
import random

import time

import matplotlib.pyplot as plt
import mediapipe as mp

# ========================================================================
@functional_datapipe("read_data_bosphorus")
class ReadData_Bosphorus(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

        self.img_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((480, 0, 1440, 1080))),
            transforms.Resize((self.img_size[0], self.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # self.base_imgs_path = "/home/cwang/Projects/thesis/dataset/bosphorus/mp96/base_images"
        # self.base_kps_path = "/home/cwang/Projects/thesis/dataset/bosphorus/mp96/base_kps"
        self.base_imgs_path = "/netscratch/cwang/datasets/bosphorus/mp96/base_images"
        self.base_kps_path = "/netscratch/cwang/datasets/bosphorus/mp96/base_kps"
        self.base_imgs_dict = {}
        self.base_heatmaps_dict = {}


    def __iter__(self):
        for data in self.source_datapipe:
            path = data[0][0]
            base_img, base_heatmaps = self.get_base_info(path)
            
            for i in range(len(data)):
                stream = data[i][1].read()

                if "rgb" in data[i][0]:
                    img = Image.open(io.BytesIO(stream))
                    img = self.transform(img)
                elif "keypoints" in data[i][0]:
                    kps = pickle.load(io.BytesIO(stream))
                    kps[:, 0] = (kps[:, 0] - 0.25) / 0.5
                    kps = kps * 256
                    heatmaps = self.kps_to_heatmaps(kps)
                elif "depth" in data[i][0]:
                    depth = pickle.load(io.BytesIO(stream))
                    depth = self.preprocess_depth(depth)

            # # Create a figure with two subplots
            # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            # axs[0, 0].imshow(base_img.permute(1, 2, 0).numpy())
            # axs[0, 0].set_title('Base Img')
            # axs[0, 0].axis('off')
            # axs[0, 1].imshow(torch.sum(base_heatmaps, dim=0), cmap='plasma')
            # axs[0, 1].set_title('Base Heatmap')
            # axs[0, 1].axis('off')
            # axs[1, 0].imshow(img.permute(1, 2, 0).numpy())
            # axs[1, 0].set_title('Img')
            # axs[1, 0].axis('off')
            # axs[1, 1].imshow(torch.sum(heatmaps, dim=0), cmap='plasma')
            # axs[1, 1].set_title('Heatmap')
            # axs[1, 1].axis('off')
            # plt.tight_layout()
            # plt.show(block=True)
                
            yield base_img, base_heatmaps, img, kps, heatmaps, depth
    
    
    def get_base_info(self, path):
        seq = path.split('/')[-2]
        seq = seq.split('.')
        seq = '.'.join(seq[:3])

        if(seq in self.base_imgs_dict):
            return self.base_imgs_dict[seq], self.base_heatmaps_dict[seq]
        else:
            base_img_path = os.path.join(self.base_imgs_path, seq+'.png')
            base_img = Image.open(base_img_path)
            base_img = self.transform(base_img)
            self.base_imgs_dict[seq] = base_img

            base_kps_path = os.path.join(self.base_kps_path, seq+'.pickle')
            with open(base_kps_path, "rb") as f:
                kps = pickle.load(f)
                kps[:, 0] = (kps[:, 0] - 0.25) / 0.5
                kps = kps * 256
            base_heatmaps = self.kps_to_heatmaps(kps)
            self.base_heatmaps_dict[seq] = base_heatmaps
                        
            return base_img, base_heatmaps
            

    def kps_to_heatmaps(self, keypoints, sigma=6):
        heatmaps = []
        for joint in keypoints:
            mu_x = int(joint[0])
            mu_y = int(joint[1])
            if(mu_x>0 and mu_y>0):
                xx, yy = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]))
                t = np.exp(-((yy - mu_y) ** 2 + (xx - mu_x) ** 2) / (2 * sigma ** 2))
            else:
                t = np.zeros((self.img_size[0], self.img_size[1]))
            heatmaps.append(torch.tensor(t))

        return torch.stack(heatmaps).float()
   

    def preprocess_depth(self, depth):
        depth = np.array([depth, depth, depth])
        depth = depth.astype('uint8')
        depth = np.moveaxis(depth, 0, -1)
        depth = Image.fromarray(depth)
        depth = self.transform(depth)
        return depth.float()
    
# ========================================================================
class DS_Bosphorus():
    def __init__(self, train=True):
        """
            1) train :      A boolean to specify if it's train or test set
        """

        self.train = train

        if(self.train):
            # self.path = '/home/cwang/Projects/thesis/dataset/bosphorus/mp96/train'
            self.path = '/netscratch/cwang/datasets/bosphorus/mp96/train'
        else:
            # self.path = '/home/cwang/Projects/thesis/dataset/bosphorus/mp96/test'
            self.path = '/netscratch/cwang/datasets/bosphorus/mp96/test'

        corrupted = ['0003.2.003.tar.gz', '0253.5.005.tar.gz', '0003.4.002.tar.gz', '0222.6.007.tar.gz', '0253.2.003.tar.gz',
                    '0248.2.005.tar.gz', '0003.2.006.tar.gz', '0222.7.006.tar.gz']
        
        # since that the dataset is so huge we will take a part of it
        self.tars = [os.path.join(self.path, file) for file in os.listdir(self.path) if not file in corrupted]
        num_tars = len(self.tars)
        self.used_tars = num_tars//25
        self.tars = self.tars[:self.used_tars]

        self.files_per_sample = 3

        
    def group_fn(self, file_path):
        """ Group files into samples based on the file name up to the first '.'.

        This will work with any dataset that follows the file naming convention of WebDataset.

        :param file_path: The path to the file for with to extract the sample name/ID.
        :return: The sample name/ID used to group the files into samples.
        """
        bn = os.path.basename(file_path[0])
        return bn.split(".")[0]

        
    def create_datapipe(self):
        if(self.train):
            random.shuffle(self.tars)
        pipe = IterableWrapper(self.tars)
        pipe = pipe.sharding_filter() \
                .open_files(mode="b").load_from_tar() \
                .groupby(group_key_fn=self.group_fn, buffer_size=self.files_per_sample, group_size=self.files_per_sample)
        if(self.train):
            pipe = pipe.shuffle()
        pipe = pipe.read_data_bosphorus()
        
        return pipe
        

if __name__ == '__main__':
    dataset = DS_Bosphorus()
    dataloader = DataLoader(dataset.create_datapipe(), batch_size=32, num_workers=16)

    for batch in dataloader:
        
        start = time.time()
    
        base_img_tensor, img_tensor, coord_tensor, heatmap_tensor, depth_tensor = batch
        print("base_img_tensor:", base_img_tensor.shape)
        print("img_tensor:", img_tensor.shape)
        print("coord_tensor:", coord_tensor.shape)
        print("heatmap_tensor:", heatmap_tensor.shape)
        print("depth_tensor:", depth_tensor.shape)
        

        end = time.time()
        print(" !!!!! TIME:", end - start)

        print("-------------------")
        