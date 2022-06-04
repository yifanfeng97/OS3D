import torch
import torch.nn as nn
from .image import MVCNN
from .voxel import VoxNet
from .pointcloud import PointNetCls


class OS3D(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img = MVCNN(n_class)
        self.model_pt = PointNetCls(n_class)
        self.model_vox = VoxNet(n_class)

    def forward(self, data, global_ft=False):
        img, pt, vox  = data
        if global_ft:
            out_img, ft_img = self.model_img(img, global_ft)
            out_pt, ft_pt = self.model_pt(pt, global_ft)
            out_vox, ft_vox = self.model_vox(vox, global_ft)
            return (out_img, out_pt, out_vox), (ft_img, ft_pt, ft_vox)
        else:
            out_img = self.model_img(img)
            out_pt = self.model_pt(pt)
            out_vox = self.model_vox(vox)
            return (out_img, out_pt, out_vox)


if __name__ == "__main__":
    pass
