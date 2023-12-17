import sparseconvnet as scn
import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class MLP(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, act_layer=nn.GELU):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or in_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            act_layer(),
            nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class TransitionUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_ch, out_ch),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(out_ch, out_ch),
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, down_xyzp, down_features, ori_xyzp, ori_features):
        down_features = self.fc1(down_features)
        ori_features = self.fc2(ori_features)

        b, n, c = ori_xyzp.shape
        _, s, _ = down_xyzp.shape
        assert s > 1

        ori_xyz, down_xyz = ori_xyzp[..., :3], down_xyzp[..., :3]
        # todo use pytorch3d ball query
        dists = torch.sum((ori_xyz[:, :, None] - down_xyz[:, None]) ** 2, dim=-1)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_features = torch.sum(knn_gather(down_features, idx) * weight.view(b, n, 3, 1), dim=2)
        return ori_features + interpolated_features


class TransitionDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_nearest: int, stride: int):
        super().__init__()
        self.stride = stride
        self.k_nearest = k_nearest
        self.mlp = nn.Sequential(
            nn.Linear(4 + in_ch, out_ch),
            SwapAxes(1, 3),
            nn.BatchNorm2d(out_ch),
            SwapAxes(1, 3),
            nn.ReLU()
        )

    def forward(self, xyzp, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, F]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_features_concat: sample points feature data, [B, S, D']
        """
        xyz = xyzp[:, :, :3]
        new_xyz, _ = sample_farthest_points(xyz, K=xyzp.shape[1] // self.stride)  # [B, out_points_num, 3]
        knn_result = knn_points(new_xyz, xyz, K=self.k_nearest)
        idx = knn_result.idx
        new_xyzp = knn_gather(xyzp, idx)[:, :, 0]

        grouped_xyzp = knn_gather(xyzp, idx)  # [B, out_points_num, k_nearest, 4]
        grouped_xyzp_norm = grouped_xyzp - new_xyzp[:, :, None]

        grouped_features = knn_gather(features, idx)  # [B, out_points_num, k_nearest, F]
        new_features_concat = torch.cat([grouped_xyzp_norm, grouped_features],
                                        dim=-1)  # [B, out_points_num, k_nearest, 4+F]
        new_features_concat = self.mlp(new_features_concat)  # [B, out_points_num, k_nearest, out_ch]
        new_features_concat = torch.max(new_features_concat, 2)[0]  # [B, out_points_num, D']
        return new_xyzp, new_features_concat


class SwapAxes(nn.Module):
    def __init__(self, dim1: int = 1, dim2: int = 2):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__()

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class PositionEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.LayerNorm(in_ch),
            nn.GELU(),
            nn.Linear(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mlp(x)


class WeightsEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear_0 = nn.Linear(in_ch, out_ch)
        self.linear_1 = nn.Linear(out_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.swap = SwapAxes(1, 2)

    def forward(self, x):
        x = self.linear_0(x)

        # BN
        b, n, k, c = x.shape
        x = x.reshape([b * n, k, c])
        x = self.swap(x)
        x = self.bn(x)
        x = self.swap(x)
        x = x.reshape([b, n, k, c])

        x = self.act(x)
        x = self.linear_1(x)
        return x


class QEncoder(nn.Module):
    def __init__(self, dim, attn_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, attn_dim),
            SwapAxes(1, 2),
            nn.BatchNorm1d(attn_dim),
            SwapAxes(1, 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)


class KEncoder(QEncoder):
    def __init__(self, dim, attn_dim):
        super().__init__(dim, attn_dim)


class SparseConv(nn.Module):
    def __init__(self, h, w, in_ch, out_ch, filter_size=3, mode='mean'):
        super().__init__()
        self.h = h
        self.w = w
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.filter_size = filter_size
        if mode == 'mean':
            mode = 4
        elif mode == 'sum':
            mode = 3
        else:
            mode = -1
        self.conv = scn.Sequential(scn.InputLayer(  # The `dimension` parameter in `scn.InputLayer` and
            # `scn.SubmanifoldConvolution` specifies the
            # dimensionality of the input data. In this case, it
            # is set to 2, indicating that the input data is
            # 2-dimensional.
            dimension=2, spatial_size=torch.LongTensor([h, w]), mode=mode),
            scn.SubmanifoldConvolution(dimension=2, nIn=in_ch + 2, nOut=out_ch,
                                       filter_size=filter_size,
                                       bias=True),
            scn.OutputLayer(out_ch))

    def forward(self, xyzp, features=None):
        B, N = xyzp.shape[:2]
        yxb = self._get_yxb(xyzp)
        pos = xyzp[..., 3][..., None]
        neg = 1 - xyzp[..., 3][..., None]
        f = [pos, neg]
        if features is not None:
            f.append(features)
        sparse = torch.cat(f, dim=-1).view(B * N, -1)
        f = self.conv([yxb, sparse]).reshape([B, N, -1])
        return f

    def _get_yxb(self, xyzp):
        B, N = xyzp.shape[:2]
        if not hasattr(self, 'b') or self.b.shape[0] != B * N or self.b.device != xyzp.device:
            self.b = torch.zeros([B * N, 1]).long().to(xyzp.device)
            for i in range(B):
                self.b[i * N:i * N + N] = i
        yx = xyzp[:, :, [1, 0]].view(-1, 2)
        yx[:, 0] = torch.round(yx[:, 0] * self.h)
        yx[:, 1] = torch.round(yx[:, 1] * self.w)
        yxb = torch.cat([yx, self.b], dim=-1).long()
        return yxb
