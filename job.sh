echo "======== Start to Train ========"
python train.py --ds_name "$1" --which_g "$2" --ema_rate "$3" --input_type "$4"
