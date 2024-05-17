srun \
  --cpus-per-task=16 \
  --gpus=1 \
  --mem="192GB" \
  --partition="RTXA6000" \
  --ntasks=1 \
  --container-mounts="/ds-av":"/ds-av","/netscratch":"/netscratch","/ds":"/ds" \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.07-py3.sqsh \
  --container-workdir="`pwd`" \
  --time=3-00:00 \
  bash job.sh "$1" "$2"
