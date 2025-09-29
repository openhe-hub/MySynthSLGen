# test.py

import os
import tqdm
import torch
from torch.utils.data import DataLoader
import argparse

# 导入所有必要的模块，和 train.py 保持一致
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
from building_components.utils import ColorPrinter, print_model_size, calculate_fid

#================================================================================================================
## Set Configuration for Testing
#================================================================================================================
parser = argparse.ArgumentParser()
# --resume_dir 现在是必须的，因为它告诉脚本从哪里加载模型
parser.add_argument('--resume_dir', type=str, required=True, help='specify the folder name of the trained model')
# --which_epoch 指定加载哪个epoch的模型，例如 '25' 或 'final'
parser.add_argument('--which_epoch', type=str, default='final', help='which epoch to test? e.g., "20" or "final"')
parser.add_argument('--ds_name', type=str, default='SynthSL', help='which dataset? ( Bosphorus / DeepFashion / Phoenix / SynthSL )')
parser.add_argument('--n_workers', type=int, default=8, help='how many workers for the datapipe?')
parser.add_argument('--batchsize_test', type=int, default=8, help='batchsize for testing')
parser.add_argument('--which_g', type=int, default=7, help='which model for the generator?')
parser.add_argument('--use_target_pose', type=bool, default=True, help='Include target pose in the discriminator input')
parser.add_argument('--input_type', type=str, default="heatmaps", help='options: heatmaps / depth / segm / normal')

# 注意：训练相关的参数在这里不再需要，例如 epochs, lr_g, lr_d 等
# 但是，为了让 GANPipe 能够正确初始化，我们可能需要保留一些参数
# 这里的参数应该与你训练时使用的参数保持一致
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for the generator')
parser.add_argument('--lr_d', type=float, default=0.00002, help='learning rate for the discriminator')
parser.add_argument('--ema_rate', type=float, default=0.9999, help='decay rate for EMA')
parser.add_argument('--hand_l1', type=bool, default=False, help='options: heatmaps / depth / segm / normal')
parser.add_argument('--which_exp', type=str, default='test', help='if add hand-masked l1')

args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------------------------------------
cprint = ColorPrinter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------------------------------------
# 加载测试数据集
if args.ds_name == "Bosphorus":
    from DS_loader.DS_Bosphorus import DS_Bosphorus
    test_dataset = DS_Bosphorus(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == "DeepFashion":
    from DS_loader.DS_DeepFashion import DS_DeepFashion
    test_dataset = DS_DeepFashion(train=False)
    n_kp = 18
elif args.ds_name == "Phoenix":
    from DS_loader.DS_Phoenix import DS_Phoenix
    test_dataset = DS_Phoenix(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == "SynthSL":
    from DS_loader.DS_SynthSL import DS_SynthSL
    test_dataset = DS_SynthSL(train=False)
    n_kp = 96 if args.input_type == "heatmaps" else 32
elif args.ds_name == 'customized':
    DATASET_ROOT_DIR = "dataset/customized_dataset" 
    from DS_loader.DS_customized import CustomDataset
    test_dataset = CustomDataset(root_dir=DATASET_ROOT_DIR, train=False)
    n_kp = 96

# ---------------------------------------------------------------------------------------------------------
# 初始化模型架构，这必须和训练时完全一样
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
# 初始化 GANPipe。它内部的逻辑应该会自动加载模型权重
# 因为我们提供了 resume_dir 参数
# 我们需要确保 GANPipe 的 __init__ 函数能够处理这种情况
# 通常它会检查 resume_dir 是否存在，如果存在就调用 load_models
args.use_pretrain = True # 强制使用预训练模型
gan = GANPipe(args, device, net_G_name, net_G, net_D, is_hand_l1=args.hand_l1)
gan.load_models(args.which_epoch) # 显式调用加载模型
net_G.eval() # 将模型设置为评估模式

#================================================================================================================
## 开始评估
#================================================================================================================
cprint("blue", f"Start testing model from {args.resume_dir} at epoch {args.which_epoch}")

ssim_list, psnr_list, pe_list, hpe_list, hssim_list = [], [], [], [], []

if args.ds_name == "DeepFashion":
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize_test, num_workers=args.n_workers)
else:
    test_loader = DataLoader(test_dataset.create_datapipe(), batch_size=args.batchsize_test, num_workers=args.n_workers)

# 创建用于保存生成图像的目录
results_dir = os.path.join(args.resume_dir, 'results', f'test_epoch_{args.which_epoch}')
os.makedirs(results_dir, exist_ok=True)
cprint("green", f"Generated images will be saved in: {results_dir}")


for i, batch in enumerate(tqdm.tqdm(test_loader)):
    cur_path = os.path.join(results_dir, f'batch_{i}.png')
    
    gan.set_input(batch)
    ssim_sublist, psnr_sublist, pe_sublist, hpe_sublist, hssim_sublist = gan.gen_results(cur_path)
    
    # 收集每个batch的结果
    ssim_list.extend(ssim_sublist)
    psnr_list.extend(psnr_sublist)
    pe_list.extend(pe_sublist)
    hpe_list.extend(hpe_sublist)
    hssim_list.extend(hssim_sublist)

# 计算所有指标的平均值
avg_ssim = sum(ssim_list) / len(ssim_list)
avg_psnr = sum(psnr_list) / len(psnr_list)
avg_pe = sum(pe_list) / len(pe_list)
avg_hpe = sum(hpe_list) / len(hpe_list)
avg_hssim = sum(hssim_list) / len(hssim_list)

# 计算 FID 分数
cprint("blue", "Calculating FID score...")
# 注意：calculate_fid 函数需要一个包含真实图像的路径和一个包含生成图像的路径
# 这里的实现可能需要调整，取决于 calculate_fid 的具体实现
# 假设它只需要生成图像的路径
fid_value = calculate_fid(results_dir)

# 将最终评估结果写入文件
eval_path = os.path.join(results_dir, 'evaluation_metrics.txt')
with open(eval_path, 'w') as f:
    f.write(f"Evaluation results for model: {args.resume_dir}, epoch: {args.which_epoch}\n")
    f.write("="*50 + "\n")
    f.write(f"SSIM: {avg_ssim}\n")
    f.write(f"PSNR: {avg_psnr}\n")
    f.write(f"PE: {avg_pe}\n")
    f.write(f"HPE: {avg_hpe}\n")
    f.write(f"HSSIM: {avg_hssim}\n")
    f.write(f"FID: {fid_value}\n")

cprint("green", f"Evaluation finished. Metrics saved to {eval_path}")
print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | FID: {fid_value:.4f}")