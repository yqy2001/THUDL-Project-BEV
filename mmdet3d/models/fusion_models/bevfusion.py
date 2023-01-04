import math

from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

        self.use_attn = False

    def activate_attn(self):
        # attn
        self.use_attn = True
        self.attn_hd = 256
        self.query_camera = nn.Linear(80, self.attn_hd)
        torch.nn.init.xavier_uniform_(self.query_camera.weight)
        # self.key_camera = nn.Linear(80, self.attn_hd)
        # torch.nn.init.xavier_uniform_(self.key_camera.weight)
        # self.value_camera = nn.Linear(80, self.attn_hd)
        # torch.nn.init.xavier_uniform_(self.value_camera.weight)
        self.proj_camera = nn.Linear(self.attn_hd, 80)
        torch.nn.init.xavier_uniform_(self.proj_camera.weight)

        # self.query_lidar = nn.Linear(256, self.attn_hd)
        # torch.nn.init.xavier_uniform_(self.query_lidar.weight)
        # self.key_lidar = nn.Linear(256, self.attn_hd)
        # torch.nn.init.xavier_uniform_(self.key_lidar.weight)
        # self.value_lidar = nn.Linear(256, self.attn_hd)
        # torch.nn.init.xavier_uniform_(self.value_lidar.weight)
        # self.proj_lidar = nn.Linear(self.attn_hd, 256)
        # torch.nn.init.xavier_uniform_(self.proj_lidar.weight)

        self.attn_drop = nn.Dropout(0.5)

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                if self.use_attn:
                    bsz, hd_cam, size_cam, size_cam = feature.shape
                    self.cam_hd = hd_cam
                    q_camera = self.query_camera(feature.permute(0, 2, 3, 1)).view(bsz, size_cam*size_cam, self.attn_hd)  # [bsz, 180 * 180, 256]
                    # k_camera = self.key_camera(feature.permute(0, 2, 3, 1)).view(bsz, size_cam*size_cam, self.attn_hd)
                    # v_camera = self.value_camera(feature.permute(0, 2, 3, 1)).view(bsz, size_cam*size_cam, self.attn_hd)
                    # q_camera = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    # k_camera = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    # v_camera = feature.reshape(feature.shape[0], feature.shape[1], -1)
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
                if self.use_attn:
                    bsz, hd_lidar, size_lidar, size_lidar = feature.shape
                    self.lidar_hd = hd_lidar
                    # q_lidar = self.query_lidar(feature.permute(0, 2, 3, 1)).view(bsz, size_lidar*size_lidar, self.attn_hd)  # [bsz, 180 * 180, attn_hd]
                    # k_lidar = self.key_lidar(feature.permute(0, 2, 3, 1)).view(bsz, size_lidar*size_lidar, self.attn_hd)
                    # v_lidar = self.value_lidar(feature.permute(0, 2, 3, 1)).view(bsz, size_lidar*size_lidar, self.attn_hd)
                    # q_lidar = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    # k_lidar = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    # v_lidar = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    lidar_feature = feature
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            if not self.use_attn:
                features.append(feature)

        if not self.training and not self.use_attn:
            # avoid OOM
            features = features[::-1]

        # features: camera [4, 80, 180, 180], lidar [4, 256, 180, 180]
        if self.use_attn:
            camera_feature = self.MSA(q_camera, lidar_feature.view(bsz, size_lidar*size_lidar, self.attn_hd), lidar_feature.view(bsz, size_lidar*size_lidar, self.attn_hd), type="cam").view(bsz, size_cam, size_cam, hd_cam).permute(0, 3, 1, 2)
            # lidar_feature = self.MSA(q_lidar, k_camera, v_camera, type="lidar").view(bsz, size_lidar, size_lidar, hd_lidar).permute(0, 3, 1, 2)

            features = [camera_feature, lidar_feature]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def MSA(self, query, key, value, num_heads=1, type="cam"):

        N, S, E = query.shape
        N, T, E = key.shape

        assert E % num_heads == 0
        head_dim = E // num_heads

        # q: [N*num_heads, S, head_dim]
        q = query.transpose(0, 1).contiguous().view(S, N * num_heads, head_dim).transpose(0, 1)
        # [N*num_heads, T, head_dim]
        k = key.transpose(0, 1).contiguous().view(T, N * num_heads, head_dim).transpose(0, 1)
        v = value.transpose(0, 1).contiguous().view(T, N * num_heads, head_dim).transpose(0, 1)

        q = q / math.sqrt(E)
        attn = torch.bmm(q, k.transpose(-2, -1))  # [N*num_heads, S, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        output = torch.bmm(attn, v)  # (N*H, S, T) x (N*H, T, E/H) = (N*H, S, E/H)
        output = output.transpose(0, 1).contiguous().view(S*N, E)  # (N*S, E)
        if type == "cam":
            output = self.proj_camera(output).view(S, N, self.cam_hd).transpose(0, 1)
        else:
            output = self.proj_lidar(output).view(S, N, self.lidar_hd).transpose(0, 1)

        return output