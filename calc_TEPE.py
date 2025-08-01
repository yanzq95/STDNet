import numpy as np

import torch
import torch.nn.functional as F

from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds


# Pre-requisite: install pytorch3d package:
# Please follow the installation guide: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

class PointsRendererWithDepth(PointsRenderer):
    """Augment PointsRenderer to output depth"""

    def __init__(self, rasterizer, compositor) -> None:
        super(PointsRendererWithDepth, self).__init__(rasterizer, compositor)

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        zbuf = fragments.zbuf.permute(0, 3, 1, 2)

        return images, F.relu(zbuf)


renderer = PointsRendererWithDepth(
    rasterizer=PointsRasterizer(),
    compositor=AlphaCompositor(background_color=(0, 0, 0)),
)


def get_delta_depth(depth, depth_prev, flow):
    # create u, v coordinate in previous frame
    h, w = depth.shape[:2]
    u_prev, v_prev = np.meshgrid(np.arange(w), np.arange(h))

    # warp to current frame with flow
    u = u_prev + flow[..., 0]
    v = v_prev + flow[..., 1]

    # camera intrinsics can be set to arbitrary
    fx = w
    fy = w
    cx = w / 2
    cy = h / 2
    intrinsics = torch.Tensor([fx, fy, cx, cy]).unsqueeze(0).float().to(device)

    # create point cloud assuming depth does not change from previous frame to current frame
    x = depth_prev * (u - cx) / fx
    y = depth_prev * (v - cy) / fy
    z = depth_prev

    pc = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=2)

    # project
    pc = torch.from_numpy(pc).unsqueeze(0).float().to(device)

    # create pc
    B = intrinsics.shape[0]
    verts = pc.reshape(B, -1, 3).contiguous()
    feat = torch.ones(B, h, w, 1).reshape(B, -1, 1).to(device)  # dummy feature to warp
    verts[..., 0] = verts[..., 0] * -1
    verts[..., 1] = verts[..., 1] * -1
    point_cloud = Pointclouds(points=verts, features=feat)

    cameras = PerspectiveCameras(
        device=device,
        principal_point=intrinsics[:, -2:],
        focal_length=intrinsics[:, :2],
        image_size=((h, w),),
        in_ndc=False,
    )

    radius = 2  # set rendering radius = 2 to avoid holes
    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=1.0
               / h
               * radius,  # The radius (in NDC units) of the disk to be rasterized.
        points_per_pixel=1,
    )
    renderer.rasterizer.cameras = cameras
    renderer.rasterizer.raster_settings = raster_settings
    feat_warp, zbuf = renderer(
        point_cloud,
        gamma=(1e-4,),
        background_color=torch.tensor(
            [0.0], dtype=torch.float32, device=device
        ),
        eps=1e-5,
    )

    # valid mask is calculated from the dummy feature warping
    valid_mask = (feat_warp > 0).float()

    # delta depth is calculated from the current frame depth and the warped depth
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(1).float().to(device)
    delta_depth = depth - zbuf

    return valid_mask, delta_depth


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

seq_len = 344  # frames-1

valid_masks = []
delta_depth_gts = []

for fidx in range(seq_len):
    # depth at previous frame
    depth_gt_prev = np.load('./VDSRDataset/TartanAir/HR_RGB/office2/P004/depth_left/{:06d}_left_depth.npy'.format(fidx)) * 100.0
    # depth at current frame
    depth_gt = np.load('./VDSRDataset/TartanAir/HR_RGB/office2/P004/depth_left/{:06d}_left_depth.npy'.format(fidx + 1)) * 100.0
    # 2D optical flow from previous frame to current frame
    flow = np.load('./VDSRDataset/TartanAir/HR_RGB/office2/P004/flow/{:06d}_{:06d}_flow.npy'.format(fidx, fidx + 1))

    valid_mask, delta_depth_gt = get_delta_depth(depth_gt, depth_gt_prev, flow)
    

    
    valid_masks.append(valid_mask[:, 0])
    delta_depth_gts.append(delta_depth_gt[:, 0])

valid_masks = torch.cat(valid_masks, dim=0)
delta_depth_gts = torch.cat(delta_depth_gts, dim=0)

delta_depth_preds = []

for fidx in range(seq_len):
    # predicted depth at previous frame
    depth_pred_prev = np.load('./Visual_Results/X16/office2_P004/{:06d}_left_depth.npy'.format(fidx)) * 100.0#[0, 0] * 10.0
    # predicted depth at current frame
    depth_pred = np.load('./Visual_Results/X16/office2_P004/{:06d}_left_depth.npy'.format(fidx + 1)) * 100.0#[0, 0] * 10.0
    # 2D optical flow from previous frame to current frame
    flow = np.load('./VDSRDataset/TartanAir/HR_RGB/office2/P004/flow/{:06d}_{:06d}_flow.npy'.format(fidx, fidx + 1))

    _, delta_depth_pred = get_delta_depth(depth_pred, depth_pred_prev, flow)

    delta_depth_preds.append(delta_depth_pred[:, 0])

delta_depth_preds = torch.cat(delta_depth_preds, dim=0)

tepe = torch.mean(torch.abs((delta_depth_gts - delta_depth_preds) * valid_masks))
print("**************************")
print('TEPE metric: ', tepe)
print("**************************")