import os
import glob
import shutil
import argparse
import random
import cv2  # OpenCV for video processing
from tqdm import tqdm

def find_video_pkl_pairs(input_dir: str) -> list:
    mp4_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    
    if not mp4_files:
        print(f"Error: No .mp4 files found in '{input_dir}'. Please check the path.")
        return []
        
    file_pairs = []
    print(f"Found {len(mp4_files)} .mp4 files. Checking for matching .pkl files...")

    for mp4_path in mp4_files:
        base_name = os.path.splitext(os.path.basename(mp4_path))[0]
        expected_pkl_path = os.path.join(input_dir, f"{base_name}.pkl")
        
        if os.path.exists(expected_pkl_path):
            file_pairs.append((mp4_path, expected_pkl_path))
        else:
            print(f"Warning: Missing .pkl file for '{mp4_path}'. This video will be skipped.")
            
    return file_pairs

def process_and_save_files(file_pairs: list, output_dir: str, img_format: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for mp4_path, pkl_path in tqdm(file_pairs, desc=f"Processing videos for '{os.path.basename(output_dir)}'"):
        base_name = os.path.splitext(os.path.basename(mp4_path))[0]
        
        # 1. Create the subdirectory for frames
        frames_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(frames_output_dir, exist_ok=True)
        
        # 2. Copy the .pkl file
        shutil.copy(pkl_path, output_dir)
        
        # 3. Extract frames from the video
        try:
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {mp4_path}. Skipping.")
                continue

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                
                # Frame filename with zero-padding (e.g., 00001.png)
                frame_filename = f"{frame_count:05d}.{img_format}"
                output_frame_path = os.path.join(frames_output_dir, frame_filename)
                cv2.imwrite(output_frame_path, frame)
                frame_count += 1
            
            cap.release()

        except Exception as e:
            print(f"An error occurred while processing {mp4_path}: {e}")


def main():
    """Main function to orchestrate the dataset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a dataset of .mp4 and .pkl files into a structured format for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="Path to the directory with your mp4 and pkl files.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where the structured dataset will be saved.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of the data to be used for training (e.g., 0.8 for 80%).")
    parser.add_argument("--img_format", type=str, default="png", choices=['png', 'jpg'], help="Image format for extracted frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting data to ensure reproducibility.")
    
    args = parser.parse_args()
    
    # --- 1. Find all matching data pairs ---
    file_pairs = find_video_pkl_pairs(args.input_dir)
    if not file_pairs:
        print("No valid video/pkl pairs found. Exiting.")
        return
    print(f"Successfully found {len(file_pairs)} matching video/pkl pairs.")
    
    # --- 2. Shuffle and split the data ---
    random.seed(args.seed)
    random.shuffle(file_pairs)
    
    split_index = int(len(file_pairs) * args.train_split)
    train_files = file_pairs[:split_index]
    test_files = file_pairs[split_index:]
    
    print("-" * 50)
    print(f"Splitting data with seed {args.seed}:")
    print(f"  Training set size: {len(train_files)} videos")
    print(f"  Test set size:     {len(test_files)} videos")
    print("-" * 50)

    # --- 3. Process and save training data ---
    if train_files:
        train_output_dir = os.path.join(args.output_dir, "train")
        process_and_save_files(train_files, train_output_dir, args.img_format)
    
    # --- 4. Process and save test data ---
    if test_files:
        test_output_dir = os.path.join(args.output_dir, "test")
        process_and_save_files(test_files, test_output_dir, args.img_format)

    print("\nDataset conversion complete!")
    print(f"Structured dataset saved at: '{args.output_dir}'")


# python3 scripts/preprocess.py --input_dir ./dataset/how2sign-zhewen --output_dir ./dataset/customized_dataset
if __name__ == "__main__":
    main()