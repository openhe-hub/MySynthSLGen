import os
import re
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp
from pytorch_fid import fid_score
from matplotlib import pyplot as plt

class OutputDirManager():
    def __init__(self, root_dir=".", output_dir="runs", resume_dir=''):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.resume_dir = resume_dir

        # Check if "runs" directory exists under root_dir
        runs_dir = os.path.join(self.root_dir, self.output_dir)
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        if self.resume_dir != '':
            self.next_subfolder_path = os.path.join(runs_dir, self.resume_dir)
            self.save_dir_models = os.path.join(self.next_subfolder_path, "models")
            self.save_dir_results = os.path.join(self.next_subfolder_path, "results")
            self.save_dir_logs = os.path.join(self.next_subfolder_path, "logs")
        else:
            # Get the list of subfolders in "runs"
            subfolders = [f for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]

            # Determine the next subfolder name
            if not subfolders:
                next_subfolder_number = 1
            else:
                subfolder_numbers = [int(re.findall(r'\d+', f)[-1]) for f in subfolders]
                next_subfolder_number = max(subfolder_numbers) + 1

            next_subfolder_name = f"exp_{next_subfolder_number}"
            self.next_subfolder_path = os.path.join(runs_dir, next_subfolder_name)

            # Create the new experiment folder
            os.makedirs(self.next_subfolder_path)
            self.save_dir_models = os.path.join(self.next_subfolder_path, "models")
            os.makedirs(self.save_dir_models)
            os.makedirs(os.path.join(self.save_dir_models, "net_G"))
            os.makedirs(os.path.join(self.save_dir_models, "net_D"))
            self.save_dir_results = os.path.join(self.next_subfolder_path, "results")
            os.makedirs(self.save_dir_results)
            os.makedirs(os.path.join(self.save_dir_results, "final"))
            self.save_dir_logs = os.path.join(self.next_subfolder_path, "logs")
        
    def get_dirs(self):
        return self.save_dir_models, self.save_dir_results, self.save_dir_logs
        

class ColorPrinter:
    def __init__(self):
        self.color_reset = "\033[0m"
        self.color_codes = {
            'red': "\033[31m",
            'green': "\033[32m",
            'yellow': "\033[33m",
            'blue': "\033[34m",
            'magenta': "\033[35m",
            'cyan': "\033[36m",
            'purple': "\033[38;5;129m",
            'olive': "\033[38;5;58m"
        }

    def __call__(self, color, message):
        color_code = self.color_codes.get(color.lower())
        if color_code:
            print(color_code + message + self.color_reset)
        else:
            print("Invalid color specified.")


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f} MB'.format(size_all_mb))


def calculate_ssim(image):
    height, width, _ = image.shape
    
    half_height = height // 2
    upper_half = image[:half_height, :]
    lower_half = image[half_height:, :]
    
    ssim_score = ssim(upper_half, lower_half, win_size=7, multichannel=True, channel_axis=2)
    
    return ssim_score


def calculate_psnr(image):
    height, width, _ = image.shape
    
    half_height = height // 2
    upper_half = image[:half_height, :]
    lower_half = image[half_height:, :]

    mse = np.mean((upper_half - lower_half) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_score = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr_score


def calculate_pe_hpe_hssims(image):
    height, width, _ = image.shape
    
    half_height = height // 2
    upper_half = image[:half_height, :]
    lower_half = image[half_height:, :]

    pose_lmks_pred, hand_lmks_pred = run_mediapipe(upper_half)
    pose_lmks_gt, hand_lmks_gt = run_mediapipe(lower_half)
        
    pe = np.sum(np.sqrt(np.sum((pose_lmks_pred - pose_lmks_gt)**2, axis=1)))
    hpe = np.sum(np.sqrt(np.sum((hand_lmks_pred - hand_lmks_gt)**2, axis=1)))

    # -------------------------------------------------------------------------
    # print("ZYZY: pred")
    hand1_pred, hand2_pred = get_hand_region(upper_half, hand_lmks_pred)
    # print("ZYZY: gt")
    hand1_gt, hand2_gt = get_hand_region(upper_half, hand_lmks_gt)

    hssims = []
    hand1_pair = (hand1_pred, hand1_gt)
    hand2_pair = (hand2_pred, hand2_gt)
    if all((img is not None) for img in hand1_pair): 
        hssims.append(calculate_ssim(np.vstack(hand1_pair)))
    if all((img is not None) for img in hand2_pair): 
        hssims.append(calculate_ssim(np.vstack(hand2_pair)))

    return pe, hpe, hssims
    

def get_hand_region(image, hand_lmks):
    hand_lmks = (hand_lmks * 256).astype(int)
    idx = 10
    hand1_region = (hand_lmks[idx][0]-30, hand_lmks[idx][1]-30, hand_lmks[idx][0]+30, hand_lmks[idx][1]+30)     # x_min, y_min, x_max, y_max
    idx = 31
    hand2_region = (hand_lmks[idx][0]-30, hand_lmks[idx][1]-30, hand_lmks[idx][0]+30, hand_lmks[idx][1]+30)
    # print("ZYZY: hand1_region", hand1_region)
    # print("ZYZY: hand2_region", hand2_region)
    if all((num > 0 and num < 256) for num in hand1_region):
        hand1_img = image[hand1_region[1]:hand1_region[3], hand1_region[0]:hand1_region[2]]
        # plt.subplot(121),plt.imshow(hand1_img,'gray'),plt.title('Hand 1')
        # print("ZYZY: hand1_img.shape", hand1_img.shape)
    else:
        hand1_img = None
        # print("invalid region!")

    if all((num > 0 and num < 256) for num in hand2_region): 
        hand2_img = image[hand2_region[1]:hand2_region[3], hand2_region[0]:hand2_region[2]]
        # plt.subplot(122),plt.imshow(hand2_img,'gray'),plt.title('Hand 2')
        # print("ZYZY: hand2_img.shape", hand2_img.shape)
    else:
        hand2_img = None
        # print("invalid region!")
    
    # plt.show(block=True)

    return hand1_img, hand2_img


def run_mediapipe(image):
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=2, enable_segmentation=True, 
                            refine_face_landmarks=True) as holistic:
        
        results = holistic.process(image)

        # ------------------------------------------------------------
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            pose_landmarks = np.array(landmarks)
        else:
            pose_landmarks = np.zeros((33, 2))
        
        # ------------------------------------------------------------
        if results.left_hand_landmarks:
            landmarks = []
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            lh_landmarks = np.array(landmarks)
        else:
            lh_landmarks = np.zeros((21, 2))

        # ------------------------------------------------------------
        if results.right_hand_landmarks:
            landmarks = []
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            rh_landmarks = np.array(landmarks)
        else:
            rh_landmarks = np.zeros((21, 2))
        # ------------------------------------------------------------
        hand_landmarks = np.concatenate((lh_landmarks, rh_landmarks), axis=0)

        return pose_landmarks, hand_landmarks
    

def calculate_fid(root_path):
    gt_path = os.path.join(root_path, "gt")
    pred_path = os.path.join(root_path, "pred")
    print("gt_path: ", gt_path)
    print("pred_path: ", pred_path)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    image_files = [f for f in os.listdir(root_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(root_path, image_file)
        split_and_save_image(image_path, gt_path, pred_path)

    fid_value = fid_score.calculate_fid_given_paths(paths=(gt_path, pred_path), device="cpu", dims=2048, batch_size=50)
    
    return fid_value
        

def split_and_save_image(image_path, gt_path, pred_path):
    # Open the image
    img = Image.open(image_path)
    basename = os.path.basename(image_path)
    
    # Get the image's width and height
    width, height = img.size
    
    # Split the upper and lower halves
    preds = img.crop((0, 0, width, height // 2))
    gts = img.crop((0, height // 2, width, height))
    
    unit = 256
    # Split the upper half into multiple small images with a width of 256 pixels
    for x in range(0, width, unit):
        small_img = preds.crop((x, 0, x + unit, height // 2))
        small_img.save(os.path.join(pred_path, basename.split('.')[0] + f"_{x//unit}.png"))
    
    # Split the lower half into multiple small images with a width of 256 pixels
    for x in range(0, width, unit):
        small_img = gts.crop((x, 0, x + unit, height // 2))
        small_img.save(os.path.join(gt_path, basename.split('.')[0] + f"_{x//unit}.png"))