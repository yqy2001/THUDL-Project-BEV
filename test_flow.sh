GPUS=8
CONFIG_DET=configs/nuscenes/det/fusion-det-flow.yaml
CONFIG_SEG=configs/nuscenes/seg/fusion-bev256d2-lss-flow.yaml
CHECKPOINT_DET=runs/run-3d91980d-e50ba4f1/latest.pth
CHECKPOINT_SEG=runs/run-3d91980d-c36bef34/latest.pth

export CUDA_HOME=/usr/local/cuda-11.1
# torchpack dist-run -np ${GPUS} python tools/test.py ${CONFIG_DET} ${CHECKPOINT_DET} --eval bbox
torchpack dist-run -np ${GPUS} python tools/test.py ${CONFIG_SEG} ${CHECKPOINT_SEG} --eval map 