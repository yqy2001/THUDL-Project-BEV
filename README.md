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
