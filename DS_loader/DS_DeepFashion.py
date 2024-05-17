import os.path
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

MISSING_VALUE = -1

class DS_DeepFashion(data.Dataset):    
    def __init__(self, train=True):
        super(DS_DeepFashion, self).__init__()
        
        self.train = train

        # root = "/home/cwang/Projects/thesis/dataset/deepfashion/wds/"
        root = "/netscratch/cwang/datasets/deepfashion/wds/"
        if(self.train):
            self.image_dir = root + "train"
            self.bone_file = root + "fasion-resize-annotation-train.csv"
            # pairLst = root + "fasion-resize-pairs-train_chenyu.csv"
            pairLst = root + "fasion-resize-pairs-train.csv"
            
        else:
            self.image_dir = root + "test"
            self.bone_file = root + "fasion-resize-annotation-test.csv"
            # pairLst = root + "fasion-resize-pairs-test_chenyu.csv"
            pairLst = root + "fasion-resize-pairs-test.csv"

        # prepare for image (image_dir), image_pair (name_pairs) and bone annotation (annotation_file)
        self.name_pairs = self.init_categories(pairLst)
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        # load image size
        self.load_size = 256

        self.old_size = (256, 176)

        # prepare for transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
       

    def __getitem__(self, index):
        # prepare for source image Xs and target image Xt
        Xs_name, Xt_name = self.name_pairs[index]
        Xs_path = os.path.join(self.image_dir, Xs_name)
        Xt_path = os.path.join(self.image_dir, Xt_name)

        Xs = Image.open(Xs_path).convert('RGB')
        Xt = Image.open(Xt_path).convert('RGB')

        Xs = F.resize(Xs, self.load_size)
        Xt = F.resize(Xt, self.load_size)

        Ps = self.obtain_bone(Xs_name)
        Xs = self.transform(Xs)
        Pt = self.obtain_bone(Xt_name)
        Xt = self.transform(Xt)
        
        # # Create a figure with two subplots
        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # axs[0, 0].imshow(Xs.permute(1, 2, 0).numpy())
        # axs[0, 0].set_title('Base Img')
        # axs[0, 0].axis('off')
        # axs[0, 1].imshow(torch.sum(Ps, dim=0), cmap='plasma')
        # axs[0, 1].set_title('Base Heatmap')
        # axs[0, 1].axis('off')
        # axs[1, 0].imshow(Xt.permute(1, 2, 0).numpy())
        # axs[1, 0].set_title('Img')
        # axs[1, 0].axis('off')
        # axs[1, 1].imshow(torch.sum(Pt, dim=0), cmap='plasma')
        # axs[1, 1].set_title('Heatmap')
        # axs[1, 1].axis('off')
        # plt.tight_layout()
        # plt.show(block=True)

        return Xs, Ps, Xt, 0, Pt, 0

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs


    def obtain_bone(self, name):
        string = self.annotation_file.loc[name]
        keypoints = self.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = self.kps_to_heatmaps(keypoints)
        return pose


    def __len__(self):
        # return len(self.name_pairs) // self.batchsize * self.batchsize
        return len(self.name_pairs)


    def name(self):
        return 'FashionDataset'
        
    #-----------------------------------------------------------------------------
    def kps_to_heatmaps(self, keypoints, sigma=6):
        heatmaps = []
        for joint in keypoints:
            joint[0] = joint[0]/self.old_size[0] * self.load_size
            joint[1] = joint[1]/self.old_size[1] * self.load_size
            mu_y = int(joint[0])
            mu_x = int(joint[1])
            if(mu_x>0 and mu_y>0):
                xx, yy = np.meshgrid(np.arange(self.load_size), np.arange(self.load_size))
                t = np.exp(-((yy - mu_y) ** 2 + (xx - mu_x) ** 2) / (2 * sigma ** 2))
            else:
                t = np.zeros((self.load_size, self.load_size))
            heatmaps.append(torch.tensor(t))

        return torch.stack(heatmaps).float()

    
    def load_pose_cords_from_strings(self, y_str, x_str):
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
