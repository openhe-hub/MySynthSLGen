import os
import cv2
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

from .utils import calculate_ssim, calculate_psnr, calculate_pe_hpe_hssims

from .DPTN import external_function
from .DPTN import base_function

from torch.utils.tensorboard import SummaryWriter
from building_components.utils import OutputDirManager
from building_components.utils import EMA

class GANPipe():
    def __init__(self, args, device, net_G_name, net_G, net_D) -> None:
        self.args = args
        self.device = device
        self.net_G_name = net_G_name

        self.net_G = net_G.to(device)
        self.net_D = net_D.to(device)

        self.ema_generator = EMA(net_G, beta=args.ema_rate)
        self.ema_discriminator = EMA(net_D, beta=args.ema_rate)

        self.optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr_g)
        self.optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr_d)

        self.gan_mode = 'vanilla'
        #self.gan_mode = 'lsgan'
        #self.gan_mode = 'hinge'
        #self.gan_mode = 'wgangp'
        
        if (self.net_G_name == "DPTN"):
            self.loss_names = ['G_l1_s', 'G_content_s', 'G_style_s', 'G_l1_t', 'G_content_t', 'G_style_t', 'G_ad_t']
        else:
            self.loss_names = ['G_l1_t', 'G_content_t', 'G_ad_t']

        self.GANloss = external_function.GANLoss(self.gan_mode).to(device)
        self.L1loss = torch.nn.L1Loss()
        self.Vggloss = external_function.VGGLoss().to(device)

        self.t_s_ratio = 0.5
        self.lambda_ad = 2.0
        self.lambda_l1 = 5.0
        self.lambda_content = 0.5
        self.lambda_style = 500

        odm = OutputDirManager(resume_dir=args.resume_dir)
        self.save_dir_models, self.save_dir_results, self.save_dir_logs = odm.get_dirs()
        self.writer = SummaryWriter(log_dir=self.save_dir_logs)

        self.use_pretrain = args.use_pretrain
        self.get_start_epoch()

        self.net_G = nn.DataParallel(net_G)
        self.net_D = nn.DataParallel(net_D)


    def get_start_epoch(self):
        n_saved_models = len([file for file in os.listdir(os.path.join(self.save_dir_models, 'net_G'))])

        ## loading partial trained models
        if(n_saved_models > 0):
            gen, disc = self.load_models_resume(self.save_dir_models)

            self.net_G.load_state_dict(gen['model_state_dict'])
            self.optimizer_G.load_state_dict(gen['optimizer_state_dict'])
            self.net_D.load_state_dict(disc['model_state_dict'])
            self.optimizer_D.load_state_dict(disc['optimizer_state_dict'])
            if 'ema_state_dict' in gen:
                self.ema_generator.shadow = gen['ema_state_dict']
            if 'ema_state_dict' in disc:
                self.ema_discriminator.shadow = disc['ema_state_dict']
            self.start_epoch = int(gen['epoch'])+1
        else:
            self.start_epoch = 1

            if self.use_pretrain:
                pretrain_model_path = os.path.join('runs', 'exp_'+self.args.which_exp, 'models')
                gen, disc = self.load_models_pretrain(pretrain_model_path, self.args.which_epoch)

                self.net_G.load_state_dict(gen['model_state_dict'])
                self.optimizer_G.load_state_dict(gen['optimizer_state_dict'])
                self.net_D.load_state_dict(disc['model_state_dict'])
                self.optimizer_D.load_state_dict(disc['optimizer_state_dict'])

                if 'ema_state_dict' in gen:
                    self.ema_generator.shadow = gen['ema_state_dict']
                if 'ema_state_dict' in disc:
                    self.ema_discriminator.shadow = disc['ema_state_dict']


        print("ZYZY: start_epoch: {}".format(self.start_epoch))
        

    def save_models(self, epoch):
        if type(epoch) == int:
            filename = 'epoch_'+str(epoch)+'.pth'
        else:
            filename = epoch
        
        filename = 'epoch_latest.pth'

        torch.save({
            'epoch':str(epoch),
            'model_state_dict': self.net_G.module.state_dict(),
            'optimizer_state_dict': self.optimizer_G.state_dict(),
            'ema_state_dict': self.ema_generator.shadow
            }, os.path.join(self.save_dir_models, 'net_G', filename))

        torch.save({
            'epoch': str(epoch),
            'model_state_dict': self.net_D.module.state_dict(),
            'optimizer_state_dict': self.optimizer_D.state_dict(),
            'ema_state_dict': self.ema_discriminator.shadow
            }, os.path.join(self.save_dir_models, 'net_D', filename))


    def load_models_resume(self, models_path):

        gen_path = os.path.join(models_path, 'net_G')
        generators =[file for file in os.listdir(gen_path) if 'final' not in file]
        last_gen_path = sorted(generators, key= lambda f: int(f.split('.')[0].split('_')[1]))[-1]

        last_gen = torch.load(os.path.join(gen_path, last_gen_path), map_location='cpu')

        disc_path = os.path.join(models_path, 'net_D')
        discriminators =[file for file in os.listdir(disc_path) if 'final' not in file]
        last_disc_path = sorted(discriminators, key= lambda f: int(f.split('.')[0].split('_')[1]))[-1]

        last_disc = torch.load(os.path.join(disc_path, last_disc_path), map_location='cpu')

        return last_gen, last_disc
    

    def load_models_pretrain(self, models_path, epoch):
        if epoch == 'final':
            gen_path = os.path.join(models_path, 'net_G', 'final.pth')
            pretrained_gen = torch.load(gen_path, map_location='cpu')

            disc_path = os.path.join(models_path, 'net_D', 'final.pth')
            pretrained_disc = torch.load(disc_path, map_location='cpu')
        else:
            gen_path = os.path.join(models_path, 'net_G', 'epoch_'+epoch+'.pth')
            pretrained_gen = torch.load(gen_path, map_location='cpu')

            disc_path = os.path.join(models_path, 'net_D', 'epoch_'+epoch+'.pth')
            pretrained_disc = torch.load(disc_path, map_location='cpu')

        return pretrained_gen, pretrained_disc


    def set_input(self, batch):
        base_img, base_heatmap, base_depth, base_segm, base_normal, img, coord, heatmap, depth, segm, normal = batch
        
        valid_input_types = ['heatmaps', 'depth', 'segm', 'normal']
        if self.args.input_type not in valid_input_types:
            raise ValueError(f"Invalid input type: {self.args.input_type}. Must be one of {valid_input_types}.")

        self.source_image = base_img.to(self.device)
        self.source_pose = {
            'heatmaps': base_heatmap.to(self.device),
            'depth': base_depth.to(self.device),
            'segm': base_segm.to(self.device),
            'normal': base_normal.to(self.device)
        }[self.args.input_type]
        
        self.target_image = img.to(self.device)
        self.target_pose = {
            'heatmaps': heatmap.to(self.device),
            'depth': depth.to(self.device),
            'segm': segm.to(self.device),
            'normal': normal.to(self.device)
        }[self.args.input_type]

        # if self.net_G_name == 'DPTN':
        #     self.target_pose = heatmap.to(self.device)
        # else:
        #     h_d_maps = torch.cat([heatmap, depth],dim=1)
        #     self.target_pose = h_d_maps.to(self.device)
            
    
    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret
    

    def forward(self):
        if (self.net_G_name == "DPTN"):
            self.fake_image_t, self.fake_image_s = self.net_G(self.source_image, self.source_pose, self.target_pose)
        else:
            self.fake_image_t = self.net_G(self.source_image, self.source_pose, self.target_pose)
    
    def get_disc_input(self, img, pose):
        if not self.args.use_target_pose:
            return img
        return torch.cat((img, pose), dim=1)

    def backward_D_basic(self, netD, real_image, fake_image, target_pose):
        # Real
        real = self.get_disc_input(real_image, target_pose)
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        fake = self.get_disc_input(fake_image, target_pose)
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        return D_loss
    

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen_t = self.backward_D_basic(self.net_D, self.target_image, self.fake_image_t, self.target_pose)
        D_loss = self.loss_dis_img_gen_t
        D_loss.backward()


    def backward_G_basic(self, fake_image, target_image, target_pose, use_d):
        # Calculate reconstruction loss
        loss_G_l1 = self.L1loss(fake_image, target_image)
        loss_G_l1 = loss_G_l1 * self.lambda_l1

        # Calculate GAN loss
        loss_G_ad = None
        if use_d:
            base_function._freeze(self.net_D)
            fake = self.get_disc_input(fake_image, target_pose)
            D_fake = self.net_D(fake)
            loss_G_ad = self.GANloss(D_fake, True, False) * self.lambda_ad

        # Calculate perceptual loss
        loss_G_content, loss_G_style = self.Vggloss(fake_image, target_image)
        loss_G_style = loss_G_style * self.lambda_style
        loss_G_content = loss_G_content * self.lambda_content

        return loss_G_l1, loss_G_ad, loss_G_style, loss_G_content


    def backward_G(self):
        base_function._unfreeze(self.net_D)

        self.loss_G_l1_t, self.loss_G_ad_t, self.loss_G_style_t, self.loss_G_content_t = self.backward_G_basic(self.fake_image_t, self.target_image, self.target_pose, use_d = True)

        if (self.net_G_name == "DPTN"):
            self.loss_G_l1_s, self.loss_G_ad_s, self.loss_G_style_s, self.loss_G_content_s = self.backward_G_basic(self.fake_image_s, self.source_image, use_d = False)
            G_loss = self.t_s_ratio*(self.loss_G_l1_t+self.loss_G_style_t+self.loss_G_content_t) + (1-self.t_s_ratio)*(self.loss_G_l1_s+self.loss_G_style_s+self.loss_G_content_s)+self.loss_G_ad_t
        else:
            G_loss = self.loss_G_l1_t + self.loss_G_content_t + self.loss_G_ad_t

        G_loss.backward()


    def optimize_parameters(self):
        
        # self.net_G.train()

        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.ema_discriminator.update()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.ema_generator.update()


    def gen_results(self, path):

        self.net_G.eval()

        if (self.net_G_name == "DPTN"):
            pred, _ = self.net_G(self.source_image, self.source_pose, self.target_pose)
        else:
            pred = self.net_G(self.source_image, self.source_pose, self.target_pose)
    
        gt = self.target_image.detach().cpu()
        pred = pred.detach().cpu()

        batch_size = pred.shape[0]
        un= UnNormalize()
        for i in range(batch_size):
            pred[i]= un(pred[i])
            gt[i]= un(gt[i])

        gt = gt.unsqueeze(4)
        pred = pred.unsqueeze(4)

        gt = gt.transpose(1,4)
        pred = pred.transpose(1,4)

        pred = pred.numpy()
        gt = gt.numpy()
        
        pred = (pred*255).astype(np.uint8)
        gt = (gt*255).astype(np.uint8)
        
        # -------------------------------------------------------
        ssim_list = []
        psnr_list = []
        pe_list = []
        hpe_list = []
        hssim_list = []
        
        im_pred = pred[0][0]
        im_gt = gt[0][0]
        im_out = np.vstack((im_pred, im_gt))

        ssim_list.append(calculate_ssim(im_out))
        psnr_list.append(calculate_psnr(im_out))
        pe, hpe, hssims = calculate_pe_hpe_hssims(im_out)
        pe_list.append(pe)
        hpe_list.append(hpe)
        hssim_list.append(hssims)

        for i in range(batch_size-1):
            
            im_pred = pred[i+1][0]
            im_gt = gt[i+1][0]
            im_pair = np.vstack((im_pred, im_gt))

            ssim_list.append(calculate_ssim(im_pair))
            psnr_list.append(calculate_psnr(im_pair))
            pe, hpe, hssims = calculate_pe_hpe_hssims(im_pair)
            pe_list.append(pe)
            hpe_list.append(hpe)
            hssim_list.append(hssims)

            im_out = np.hstack((im_out, im_pair))
            
        im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)

        if(cv2.imwrite(path, im_out)):
            print(">>>>>>> saved image to ", path)
        else:
            print('!!!!!!! failed to save images !!!!!!!')

        hpe_list = [num for num in hpe_list if num != 0]
        hssim_list = [item for sublist in hssim_list for item in sublist]

        return ssim_list, psnr_list, pe_list, hpe_list, hssim_list
        

class UnNormalize(object):
    def __init__(self,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor