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

import time

import matplotlib.pyplot as plt
import mediapipe as mp

# ========================================================================
@functional_datapipe("read_data_phoenix")
class ReadData_Phoenix(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

        self.img_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # self.base_imgs_tar = '/home/cwang/Projects/thesis/dataset/phoenix-2014-T.v3/chenyu/base_imgs_tar.tar.gz'
        # self.base_kps_tar = '/home/cwang/Projects/thesis/dataset/phoenix-2014-T.v3/chenyu/base_kps_tar.tar.gz'
        self.base_imgs_tar = '/netscratch/alnaqish/datasets/phoenix-2014-T.v3/mp96/base_imgs_tar.tar.gz'
        self.base_kps_tar = '/netscratch/alnaqish/datasets/phoenix-2014-T.v3/mp96/base_kps_tar.tar.gz'
        self.base_imgs_dict, self.base_kps_dict = self.get_base_info()


    def __iter__(self):
        for data in self.source_datapipe:
            path = data[0][0]
            basename = os.path.basename(path)
            seq_number = basename.split('.')[0]
            base_img = self.base_imgs_dict[seq_number]
            base_kps = self.base_kps_dict[seq_number]
            base_heatmaps = self.kps_to_heatmaps(base_kps)
            
            for i in range(len(data)):
                stream = data[i][1]

                if "rgb" in data[i][0]:
                    img = Image.open(io.BytesIO(stream))
                    img = self.transform(img)
                elif "keypoints" in data[i][0]:
                    kps = pickle.load(io.BytesIO(stream)) * 256
                    heatmaps = self.kps_to_heatmaps(kps)
                elif "depth" in data[i][0]:
                    depth = Image.open(io.BytesIO(stream))

                    if(len(depth.size) == 2):
                        depth = depth.convert('RGB')

                    depth = self.transform(depth)
                elif "segm" in data[i][0]:
                    pass

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

            yield base_img, base_heatmaps, 0, 0, 0, img, kps, heatmaps, 0, 0, 0
    

    def get_base_info(self):
        """
            returns a dictionary with the keys are the sequence numbers and the values are the base images for each sequence
        """
        tar = tarfile.open(self.base_imgs_tar)
        base_imgs_dict = {}
        for name, member in zip(tar.getnames(), tar.getmembers()):
            seq = name.split('.')[0]
            img = tar.extractfile(member)
            img = img.read()
            img = Image.open(io.BytesIO(img))
            base_imgs_dict[seq] = self.transform(img)
        tar.close()

        tar = tarfile.open(self.base_kps_tar)
        base_kps_dict = {}
        for name, member in zip(tar.getnames(), tar.getmembers()):
            seq = name.split('.')[0]
            kps = tar.extractfile(member)
            kps = kps.read()
            kps = pickle.load(io.BytesIO(kps)) * 256
            base_kps_dict[seq] = kps
        tar.close()

        return base_imgs_dict, base_kps_dict
    

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
    

# ========================================================================
class DS_Phoenix():
    def __init__(self, train=True):
        """
            1) train :      A boolean to specify if it's train or test set
        """

        self.train = train

        if(self.train):
            self.path = "/netscratch/alnaqish/datasets/phoenix-2014-T.v3/mp96/train.tar.gz"
        else:
            self.path = "/netscratch/alnaqish/datasets/phoenix-2014-T.v3/mp96/test.tar.gz"

        # self.path = "/home/cwang/Projects/thesis/dataset/phoenix-2014-T.v3/chenyu/chenyu.tar.gz"

        tar = tarfile.open(self.path)
        self.files = list(zip(tar.getnames(), tar.getmembers()))
        print('Tar file is loaded successfully.')
        self.data = [(f[0], tar.extractfile(f[1]).read()) for f in self.files]
        print('Binary streams of the files in Tar are read successfully')

        self.files_per_sample = 2
    

    def group_fn(self, source_datapipe):
        path = source_datapipe[0]
        basename = os.path.basename(path)
        splitted = basename.split('.')
        return splitted[0] + "." + splitted[1]


    def create_datapipe(self):
        pipe = IterableWrapper(self.data)
        pipe = pipe.groupby(group_key_fn=self.group_fn, buffer_size=self.files_per_sample, group_size=self.files_per_sample)
        if(self.train):
            pipe = pipe.shuffle()
        pipe = pipe.sharding_filter() \
                .read_data_phoenix()
                    
        return pipe
        
        