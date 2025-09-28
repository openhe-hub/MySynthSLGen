import os
import tqdm

import torch
from torch.utils.data import DataLoader

from building_components.Baseline.PatchGAN import define_G
from building_components.Baseline.UNetGenerator import UNetGenerator
from building_components.ViT.ViTGenerator_v2 import ViTGenerator
from building_components.Swin.SwinGenerator import SwinTransformer
from building_components.Swin.SwinGenerator_v2 import SwinTransformerV2
from building_components.Swin.SwinGenerator_v2_concat3 import SwinTransformerV2_concat3
from building_components.Swin.SwinGenerator_v2_concat4 import SwinTransformerV2_concat4
from building_components.Swin.SwinGenerator_v2_concat5 import SwinTransformerV2_concat5
from building_components.StyleSwin.StyleSwinGenerator import Generator
from building_components.DPTN.networks import DPTNGenerator, ResDiscriminator
from building_components.GANPipe import GANPipe
from building_components.utils import ColorPrinter, print_model_size
from building_components.utils import calculate_fid
import argparse

#================================================================================================================
## Set Configuration
#================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--resume_dir', type=str, default='', help='specify the folder name if you want to continue that training')
parser.add_argument('--epochs', type=int, default=30, help='how many epochs in total?')
parser.add_argument('--ds_name', type=str, default='SynthSL', help='which dataset? ( Bosphorus / DeepFashion / Phoenix / SynthSL )')
parser.add_argument('--n_workers', type=int, default=8, help='how many workers for the datapipe?')
parser.add_argument('--batchsize_train', type=int, default=8, help='batchsize for training')
parser.add_argument('--batchsize_test', type=int, default=8, help='batchsize for testing')
parser.add_argument('--which_g', type=int, default=7, help='which model for the generator?')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for the generator')
parser.add_argument('--lr_d', type=float, default=0.00002, help='learning rate for the discriminator')
parser.add_argument('--use_pretrain', type=bool, default=False, help='if use pretrained model or not?')
parser.add_argument('--which_exp', type=str, default='10', help='which exp?')
parser.add_argument('--which_epoch', type=str, default='final', help='which epoch?')
parser.add_argument('--use_target_pose', type=bool, default=True, help='Include target pose in the discriminator input')
parser.add_argument('--ema_rate', type=float, default=0.9999, help='decay rate for EMA')
parser.add_argument('--input_type', type=str, default="heatmaps", help='options: heatmaps / depth / segm / normal')
parser.add_argument('--hand_l1', type=bool, default=False, help='options: heatmaps / depth / segm / normal')

args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------------------------------------
cprint = ColorPrinter()

num_gpus = torch.cuda.device_count()
device = torch.device("cuda" if num_gpus > 0 else "cpu")

# ---------------------------------------------------------------------------------------------------------
# n_kp is the number of channels in the condition image
if args.ds_name == "Bosphorus":
    from DS_loader.DS_Bosphorus import DS_Bosphorus
    train_dataset = DS_Bosphorus(train=True)
    test_dataset = DS_Bosphorus(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == "DeepFashion":
    from DS_loader.DS_DeepFashion import DS_DeepFashion
    train_dataset = DS_DeepFashion(train=True)
    test_dataset = DS_DeepFashion(train=False)
    n_kp = 18
elif args.ds_name == "Phoenix":
    from DS_loader.DS_Phoenix import DS_Phoenix
    train_dataset = DS_Phoenix(train=True)
    test_dataset = DS_Phoenix(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == "SynthSL":
    from DS_loader.DS_SynthSL import DS_SynthSL
    train_dataset = DS_SynthSL(train=True)
    test_dataset = DS_SynthSL(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == 'customized':
    DATASET_ROOT_DIR = "dataset/customized_dataset" 
    from DS_loader.DS_customized import CustomDataset
    train_dataset = CustomDataset(root_dir=DATASET_ROOT_DIR, train=True)
    test_dataset = CustomDataset(root_dir=DATASET_ROOT_DIR, train=False)
    n_kp = 96

# ---------------------------------------------------------------------------------------------------------
net_G_dict = {
    0: ("UNet",             UNetGenerator(2*n_kp+3, heatmap_channels=n_kp, device_encoder=device, device_decoder=device)),
    1: ("ResNet",           define_G(2*n_kp+3, 3, 64, 'resnet_9blocks')),
    2: ("ViT",              ViTGenerator(2*n_kp+3)),
    3: ("Swin",             SwinTransformer(img_size=256, window_size=8, in_chans=2*n_kp+3)),
    4: ("Swin_v2",          SwinTransformerV2(img_size=256, window_size=8, in_chans=2*n_kp+3)),
    5: ("Swin_v2_concat3",   SwinTransformerV2_concat3(n_kp=n_kp, img_size=256, window_size=8, in_chans=2*n_kp+3)),
    6: ("Swin_v2_concat4",   SwinTransformerV2_concat4(n_kp=n_kp, img_size=256, window_size=8, in_chans=3)),
    7: ("Swin_v2_concat5",   SwinTransformerV2_concat5(n_kp=n_kp, img_size=256, window_size=8, in_chans=96*2+3)),
    8: ("StyleSwin",        Generator(size=256, n_kp=n_kp)),
    9: ("DPTN",             DPTNGenerator(image_nc=3, pose_nc=n_kp, ngf=64, img_f=512, layers=3, norm='instance', activation='LeakyReLU', 
                                      use_spect=False, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2))
}
net_G_name = net_G_dict[args.which_g][0]
net_G = net_G_dict[args.which_g][1]
input_nc = 3
if args.use_target_pose:
    input_nc += n_kp if args.input_type == 'heatmaps' else 3
net_D = ResDiscriminator(input_nc=input_nc, ndf=32, img_f=128, layers=3, norm='none', activation='LeakyReLU', use_spect=True)
print_model_size(net_G)

# ---------------------------------------------------------------------------------------------------------
gan = GANPipe(args, device, net_G_name, net_G, net_D, is_hand_l1=args.hand_l1)

#================================================================================================================
## Formally start training
#================================================================================================================
cprint("blue", '_'*50)
cprint("blue", 'Formally start training')

for epoch in range(gan.start_epoch, args.epochs+1):

    cprint("blue", '_'*50)
    cprint("blue", 'EPOCH ' + str(epoch))
    cprint("blue", '_'*50)

    epoch_total_losses = {}
    if (net_G_name == "DPTN"):
        epoch_total_losses['G_l1_s'] = 0
        epoch_total_losses['G_content_s'] = 0
        epoch_total_losses['G_style_s'] = 0
        epoch_total_losses['G_style_t'] = 0
    epoch_total_losses['G_l1_t'] = 0
    epoch_total_losses['G_content_t'] = 0
    epoch_total_losses['G_ad_t'] = 0

    if args.ds_name == "DeepFashion":
        dataloader = DataLoader(train_dataset, batch_size=args.batchsize_train, num_workers=args.n_workers)
    else:
        dataloader = DataLoader(train_dataset.create_datapipe(), batch_size=args.batchsize_train, num_workers=args.n_workers)
    pbar = tqdm.tqdm(dataloader)
    
    for batch in pbar:
        #-----------------------------------------------------------------------------------------------------------
        ## training model
        gan.set_input(batch)
        gan.optimize_parameters()
        
        losses = gan.get_current_errors()
        if (net_G_name == "DPTN"):
            epoch_total_losses['G_l1_s'] += losses['G_l1_s']
            epoch_total_losses['G_content_s'] += losses['G_content_s']
            epoch_total_losses['G_style_s'] += losses['G_style_s']
            epoch_total_losses['G_style_t'] += losses['G_style_t']
        epoch_total_losses['G_l1_t'] += losses['G_l1_t']
        epoch_total_losses['G_content_t'] += losses['G_content_t']
        epoch_total_losses['G_ad_t'] += losses['G_ad_t']

        formatted_epoch_total_losses = " | ".join([f"{key}: {value:.2f}" for key, value in epoch_total_losses.items()])
        pbar.set_description(formatted_epoch_total_losses)
        
    #-----------------------------------------------------------------------------------------------------------
    ## for every 5 epochs, we save the models and the results
    if(epoch%1 == 0):         
        # save_models(net_G, net_D, optimizer_G, optimizer_D, save_dir_models, epoch)
        gan.save_models(epoch)
        
        #------------------------------------
        # generate results
        cur_path = os.path.join(gan.save_dir_results, 'epoch_'+str(epoch)+'.png')

        if args.ds_name == "DeepFashion":
            test_loader = DataLoader(test_dataset, batch_size=args.batchsize_test, num_workers=args.n_workers)
        else:
            test_loader = DataLoader(test_dataset.create_datapipe(), batch_size=args.batchsize_test, num_workers=args.n_workers)
        batch = next(iter(test_loader))
        
        gan.set_input(batch)
        ssim_sublist, psnr_sublist, pe_sublist, hpe_sublist, hssim_sublist = gan.gen_results(cur_path)
        
        avg_ssim = sum(ssim_sublist) / len(ssim_sublist)
        avg_psnr = sum(psnr_sublist) / len(psnr_sublist)
        avg_pe = sum(pe_sublist) / len(pe_sublist)
        avg_hpe = sum(hpe_sublist) / len(hpe_sublist)
        
        print("avg_ssim: {}".format(avg_ssim))
        print("avg_psnr: {}".format(avg_psnr))
        print("avg_pe: {}".format(avg_pe))
        print("avg_hpe: {}".format(avg_hpe))
        print("hssim_list: {}".format(hssim_sublist))

        gan.writer.add_scalar("eval/ssim", avg_ssim, global_step=epoch)
        gan.writer.add_scalar("eval/psnr", avg_psnr, global_step=epoch)
        gan.writer.add_scalar("eval/pe", avg_pe, global_step=epoch)
        gan.writer.add_scalar("eval/hpe", avg_hpe, global_step=epoch)



#================================================================================================================
## After training
#================================================================================================================
gan.save_models("final.pth")
        
## generate results
ssim_list = []
psnr_list = []
pe_list = []
hpe_list = []
hssim_list = []

if args.ds_name == "DeepFashion":
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize_test, num_workers=args.n_workers)
else:
    test_loader = DataLoader(test_dataset.create_datapipe(), batch_size=args.batchsize_test, num_workers=args.n_workers)

for i, batch in enumerate(test_loader):
    cur_path = os.path.join(gan.save_dir_results, 'final', 'batch_'+str(i)+'.png')
    
    gan.set_input(batch)
    ssim_sublist, psnr_sublist, pe_sublist, hpe_sublist, hssim_sublist = gan.gen_results(cur_path)
    ssim_list.append(ssim_sublist)
    psnr_list.append(psnr_sublist)
    pe_list.append(pe_sublist)
    hpe_list.append(hpe_sublist)
    hssim_list.append(hssim_sublist)

# Flatten the list of lists using list comprehension
ssim_list = [item for sublist in ssim_list for item in sublist]
psnr_list = [item for sublist in psnr_list for item in sublist]
pe_list = [item for sublist in pe_list for item in sublist]
hpe_list = [item for sublist in hpe_list for item in sublist]
hssim_list = [item for sublist in hssim_list for item in sublist]

avg_ssim = sum(ssim_list) / len(ssim_list)
avg_psnr = sum(psnr_list) / len(psnr_list)
avg_pe = sum(pe_list) / len(pe_list)
avg_hpe = sum(hpe_list) / len(hpe_list)
avg_hssim = sum(hssim_list) / len(hssim_list)

cur_path = os.path.join(gan.save_dir_results, 'final')
fid_value = calculate_fid(cur_path)

eval_path = os.path.join(gan.save_dir_results, 'final', 'evaluation.txt')
with open(eval_path, 'w') as f:
    f.write(f"SSIM: {avg_ssim}\n")
    f.write(f"PSNR: {avg_psnr}\n")
    f.write(f"PE: {avg_pe}\n")
    f.write(f"HPE: {avg_hpe}\n")
    f.write(f"HSSIM: {avg_hssim}\n")
    f.write(f"FID: {fid_value}\n")

gan.writer.flush()
gan.writer.close()
