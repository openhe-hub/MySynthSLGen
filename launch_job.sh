srun \
  --cpus-per-task=16 \
  --gpus=1 \
  --mem="256GB" \
  --partition="RTXA6000" \
  --ntasks=1 \
  --container-mounts="/ds-av":"/ds-av","/netscratch":"/netscratch","/ds":"/ds" \
  --container-image=/netscratch/alnaqish/images/upd_image.sqsh \
  --container-workdir="`pwd`" \
  --time=3-00:00 \
  bash job.sh "$1" "$2" "$3" "$4"