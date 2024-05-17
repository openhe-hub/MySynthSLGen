echo "======== Start to Install the Dependencies =========="
pip install timm torchsummary torch_geometric scikit-image
pip install opencv-contrib-python==4.7.0.72
pip install mediapipe
pip install pytorch_fid

echo "======== Start to Train ========"
python train.py --ds_name "$1" --which_g "$2"
