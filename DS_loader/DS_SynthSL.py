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
@functional_datapipe("read_data_synthsl")
class ReadData_SynthSL(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

        self.img_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # self.base_imgs_tar = '/home/cwang/Projects/thesis/dataset/synthsl/v1.0/base_imgs_tar.tar'
        # self.base_kps_tar = '/home/cwang/Projects/thesis/dataset/synthsl/v1.0/base_kps_tar.tar'
        # self.base_metas_tar = '/home/cwang/Projects/thesis/dataset/synthsl/v1.0/base_metas_tar.tar'
        self.base_imgs_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_imgs_tar.tar'
        self.base_kps_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_kps_tar.tar'
        self.base_metas_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_metas_tar.tar'
        self.base_depths_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_depths_tar.tar'
        self.base_segms_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_segms_tar.tar'
        self.base_normals_tar = '/ds-av/internal_datasets/synthsl/wds/v1.0/base_normals_tar.tar'

        self.use_meta = True
        self.useful_keys = [0, 9, 12, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 66, 
                            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 
                            89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 
                            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 124, 125, 126]

        self.base_imgs_dict, self.base_kps_dict, self.base_depths_dict, self.base_segms_dict, self.base_normals_dict = self.get_base_info()


    def __iter__(self):
        for data in self.source_datapipe:
            path = data[0][0]
            basename = os.path.basename(path)
            seq_number = basename.split('.')[0]
            base_img = self.base_imgs_dict[seq_number]
            base_kps = self.base_kps_dict[seq_number]
            base_heatmaps = self.kps_to_heatmaps(base_kps)
            base_depth = self.base_depths_dict[seq_number]
            base_segm = self.base_segms_dict[seq_number]
            base_normal = self.base_normals_dict[seq_number]

            for i in range(len(data)):
                stream = data[i][1]

                if "rgb" in data[i][0]:
                    img = Image.open(io.BytesIO(stream))
                    img = self.transform(img)
                elif "keypoints" in data[i][0]:
                    kps = pickle.load(io.BytesIO(stream)) * 256
                    heatmaps = self.kps_to_heatmaps(kps)
                # elif "meta" in data[i][0]:
                #     meta = pickle.load(io.BytesIO(stream))
                #     kps = np.array(meta['full_body_2d'])
                #     kps = kps[self.useful_keys]
                #     kps = kps * 0.5
                #     heatmaps = self.kps_to_heatmaps(kps)
                elif "depth" in data[i][0]:
                    depth = Image.open(io.BytesIO(stream))

                    if(len(depth.size) == 2):
                        depth = depth.convert('RGB')

                    depth = self.transform(depth)
                elif "segm" in data[i][0]:
                    segm = Image.open(io.BytesIO(stream))
                    segm = self.transform(segm)
                elif "normal" in data[i][0]:
                    normal = Image.open(io.BytesIO(stream))
                    normal = self.transform(normal)

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

            yield base_img, base_heatmaps, base_depth, base_segm, base_normal, img, kps, heatmaps, depth, segm, normal
    

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

        tar = tarfile.open(self.base_depths_tar)
        base_depths_dict = {}
        for name, member in zip(tar.getnames(), tar.getmembers()):
            seq = name.split('.')[0]
            depth = tar.extractfile(member)
            depth = depth.read()
            depth = Image.open(io.BytesIO(depth))
            if(len(depth.size) == 2):
                depth = depth.convert('RGB')
            base_depths_dict[seq] = self.transform(depth)
        tar.close()

        tar = tarfile.open(self.base_segms_tar)
        base_segms_dict = {}
        for name, member in zip(tar.getnames(), tar.getmembers()):
            seq = name.split('.')[0]
            segm = tar.extractfile(member)
            segm = segm.read()
            segm = Image.open(io.BytesIO(segm))
            base_segms_dict[seq] = self.transform(segm)
        tar.close()

        tar = tarfile.open(self.base_normals_tar)
        base_normals_dict = {}
        for name, member in zip(tar.getnames(), tar.getmembers()):
            seq = name.split('.')[0]
            normal = tar.extractfile(member)
            normal = normal.read()
            normal = Image.open(io.BytesIO(normal))
            base_normals_dict[seq] = self.transform(normal)
        tar.close()

        # tar = tarfile.open(self.base_metas_tar)
        # base_kps_dict = {}
        # for name, member in zip(tar.getnames(), tar.getmembers()):
        #     seq = name.split('.')[0]
        #     meta = tar.extractfile(member)
        #     meta = meta.read()
        #     meta = pickle.load(io.BytesIO(meta))
        #     kps = np.array(meta['full_body_2d'])
        #     kps = kps[self.useful_keys]
        #     kps = kps * 0.5
        #     base_kps_dict[seq] = kps
        # tar.close()

        return base_imgs_dict, base_kps_dict, base_depths_dict, base_segms_dict, base_normals_dict
    

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
class DS_SynthSL():
    def __init__(self, train=True):
        """
            1) train :      A boolean to specify if it's train or test set
        """

        self.train = train

        if(self.train):
            self.path = "/ds-av/internal_datasets/synthsl/wds/v1.0/training.tar"
        else:
            self.path = "/ds-av/internal_datasets/synthsl/wds/v1.0/test.tar"

        # self.path = "/home/cwang/Projects/thesis/dataset/synthsl/v1.0/chenyu.tar"

        tar = tarfile.open(self.path)
        self.files = list(zip(tar.getnames(), tar.getmembers()))
        print('Tar file is loaded successfully.')
        self.data = [(f[0], tar.extractfile(f[1]).read()) for f in self.files]
        print('Binary streams of the files in Tar are read successfully')

        self.files_per_sample = 6
    

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
                .read_data_synthsl()
                    
        return pipe
        

if __name__ == '__main__':
    dataset = DS_SynthSL()
    dataloader = DataLoader(dataset.create_datapipe(), batch_size=32, num_workers=16)

    for batch in dataloader:
        
        start = time.time()
    
        base_img_tensor, img_tensor, coord_tensor, heatmap_tensor, depth_tensor, segm_tensor = batch
        print("base_img_tensor:", base_img_tensor.shape)
        print("img_tensor:", img_tensor.shape)
        print("coord_tensor:", coord_tensor.shape)
        print("heatmap_tensor:", heatmap_tensor.shape)
        print("depth_tensor:", depth_tensor.shape)
        print("segm_tensor:", segm_tensor.shape)
        

        end = time.time()
        print(" !!!!! TIME:", end - start)

        print("-------------------")
        