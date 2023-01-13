# BEVFusion
This is an extension of [BEVFusion](https://arxiv.org/abs/2205.13542) with many fusion methods implemented and evaluated on nuScenes dataset.

## Usage
### Requirements
As described [here](https://github.com/mit-han-lab/bevfusion), the code is built with following libraries:
- Python >= 3.8, <3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0
- PyTorch >= 1.9, <= 1.10.2
- tqdm
- torchpack
- mmcv = 1.4.0
- mmdetection = 2.20.0
- nuscenes-dev-kit

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

Or alternatively, simply run 

```bash
conda env create -n <name> -f bevmit.yaml
```

### Running

Prepare dataset and pretrained models on AIR-slurm before running:

```shell
ln -s /home/aidrive/lpf/bevfusion/code/bevfusion/pretrained ./pretrained

ln -s /home/DISCOVER/lpf/bevfusion/dataset/ ./data

torchpack dist-run -np 2 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth
```

### Config

Change batch size per gpu to accommodate different GPU memory: 

>  ./configs/nuscenes/default.yaml   `samples_per_gpu`

### Evaluation

For BEVFusion with alignment and fusion module (BEVFusion-flow), if evaluated on 3D object detection task, please run:

```bash
torchpack dist-run -np ${GPUS} python tools/test.py configs/nuscenes/det/fusion-det-flow.yaml ${CHECKPOINT} --eval bbox
```

Otherwise, if evaluated on segmentation task, please run:

```bash
torchpack dist-run -np ${GPUS} python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss-flow.yaml ${CHECKPOINT} --eval map 
```

### Training

For BEVFusion with alignment and fusion module (BEVFusion-flow), if trained on 3D object detection task, please run:

```bash
torchpack dist-run -np ${GPUS} python tools/train.py configs/nuscenes/det/fusion-det-flow.yaml
```

Otherwise, if trained on segmentation task, please run:

```bash
torchpack dist-run -np ${GPUS} python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss-flow.yaml
```

For BEVFusion with cross-attention, the modifications are in the `./mmdet3d/models/fusion_models/bevfusion.py` file. To train the model, please first change the use-attn flag to `True` (line 26 of `tools/train.py`) to activate cross-attention layers, then run:

```shell
torchpack dist-run -np ${GPUS} python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth
```

