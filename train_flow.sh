GPUS=4
CONFIG_DET=configs/nuscenes/det/fusion-det-flow.yaml
CONFIG_SEG=configs/nuscenes/seg/fusion-bev256d2-lss-flow.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,4
export CUDA_HOME=/usr/local/cuda-11.1

torchpack dist-run -np ${GPUS} python tools/train.py ${CONFIG_DET} --load_from runs/run-3d91980d-48d0e7c5/latest.pth
# torchpack dist-run -np ${GPUS} python tools/train.py ${CONFIG_SEG} ${CHECKPOINT_FILE} ${SEG_PY_ARGS} #--out results.pkl