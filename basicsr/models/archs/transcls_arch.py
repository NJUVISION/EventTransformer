from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional
import matplotlib.pyplot as plt
from pytorch3d.ops import knn_points, knn_gather, ball_query, sample_farthest_points
from timm.models.layers import DropPath

from basicsr.models.archs.arch_util import TransitionDown, MLP, PositionEncoder, SparseConv


class EventAttention(nn.Module):
    def __init__(self, dim, attn_dim, k_nearest=16, h=128, w=128, conv_kernel_size=3, global_step=8):
        super().__init__()
        self.k_nearest = k_nearest
        self.attn_scale = sqrt(attn_dim)
        self.h, self.w = h, w
        self.conv_kernel_size = conv_kernel_size
        self.global_step = global_step

        self.local_qkv = nn.Linear(dim, attn_dim * 3)
        self.local_pe = PositionEncoder(4, attn_dim)
        self.local_fc = nn.LayerNorm(attn_dim)
        proj_dim = attn_dim

        if self.conv_kernel_size > 0:
            self.conv_qkv = SparseConv(
                in_ch=dim, out_ch=attn_dim * 3, h=h, w=w, filter_size=conv_kernel_size)
            self.conv_pe = PositionEncoder(2, attn_dim)
            self.conv_fc = nn.LayerNorm(attn_dim)
            proj_dim += attn_dim

        if self.global_step > 0:
            self.global_qkv = nn.Linear(dim, attn_dim * 3)
            self.global_pe = PositionEncoder(4, attn_dim)
            self.global_fc = nn.LayerNorm(attn_dim)
            proj_dim += attn_dim

        self.proj = MLP(in_ch=proj_dim, hidden_ch=dim, out_ch=dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xyzp, features):
        attn = self.local_attn(xyzp, features)
        if self.conv_kernel_size > 0:
            conv_attn = self.conv_attn(xyzp, features)
            attn = torch.cat([attn, conv_attn], dim=-1)
        if self.global_step > 0:
            global_attn = self.global_attn(xyzp, features)
            attn = torch.cat([attn, global_attn], dim=-1)

        attn = self.proj(attn)
        return attn

    def local_attn(self, xyzp, features):
        # local position encoding
        xyz = xyzp[:, :, :3]
        idx = knn_points(xyz, xyz, K=self.k_nearest).idx
        # [B, N, k_nearest, attn_dim]
        pos_enc = self.local_pe(xyzp[:, :, None] - knn_gather(xyzp, idx))
        # local attention
        qkv = self.local_qkv(features)
        c = qkv.shape[-1] // 3
        q, k, v = qkv[..., :c], qkv[..., c:2 *
                                    c], qkv[..., 2 * c:]  # [B, N, attn_dim]
        k, v = knn_gather(k, idx), knn_gather(
            v, idx)  # [B, N, k_nearest, attn_dim]
        local_attn = self.local_fc(
            q[:, :, None, :] - k + pos_enc) / self.attn_scale
        local_attn = self.softmax(local_attn)
        local_attn = local_attn * (v + pos_enc)
        local_attn = torch.sum(local_attn, dim=2)  # [B, N, attn_dim]
        return local_attn

    def conv_attn(self, xyzp, features):
        # conv position encoding
        xyz = xyzp[..., :3].clone().detach()
        xyz[..., 2] = 0
        temp = ball_query(xyz, xyz, radius=5 / self.h,
                          K=self.k_nearest)  # [B, N, k_nearest]
        idx = temp.idx
        # mask = torch.where(idx == -1,
        #    torch.zeros_like(idx), torch.ones_like(idx))
        idx = torch.where(idx == -1, self._get_self_idx(idx),
                          idx)  # replace -1 idx to self idx
        xy = xyzp[..., :2]
        # [B, N, k_nearest, attn_dim]
        pos_enc = self.conv_pe(xy[:, :, None] - knn_gather(xy, idx))
        # has a bug when xy.shape == [b, 8] and idx = [..., 7, 7, 7]
        # conv attention
        qkv = self.conv_qkv(xyzp, features)

        c = qkv.shape[-1] // 3
        q, k, v = qkv[..., :c], qkv[..., c:2 *
                                    c], qkv[..., 2 * c:]  # [B, N, attn_dim]
        k, v = knn_gather(k, idx), knn_gather(
            v, idx)  # [B, N, k_nearest, attn_dim]
        conv_attn = self.conv_fc(
            q[:, :, None, :] - k + pos_enc) / self.attn_scale
        conv_attn = self.softmax(conv_attn)
        conv_attn = conv_attn * (v + pos_enc)
        # conv_attn = conv_attn * mask[:, :, :, None] + (1 - mask[:, :, :, None]) * (-100)
        conv_attn = torch.sum(conv_attn, dim=2)  # [B, N, attn_dim]
        return conv_attn

    def global_attn(self, xyzp, features):
        # global position encoding
        xyz = xyzp[:, :, :3]
        down_xyz, down_idx = sample_farthest_points(
            xyz, K=xyzp.shape[1] // self.global_step)  # [B, global_points_num]
        down_xyzp = knn_gather(xyzp, down_idx[:, :, None])[
            :, :, 0, :]  # [B, global_points_num]
        # [B, global_points_num, k_nearest]
        pair_idx = knn_points(down_xyz, xyz, K=self.k_nearest).idx
        inv_pair_idx = knn_points(
            xyz, down_xyz, K=self.k_nearest).idx  # [B, N, k_nearest]
        # [B, N, k_nearest, attn_dim]
        pos_enc = self.global_pe(
            xyzp[:, :, None] - knn_gather(down_xyzp, inv_pair_idx))

        # global attention
        qkv = self.global_qkv(features)
        c = qkv.shape[-1] // 3
        q, k, v = qkv[..., :c], qkv[..., c:2 *
                                    c], qkv[..., 2 * c:]  # [B, N, attn_dim]
        # [B, global_points_num, k_nearest, attn_dim]
        k, v = knn_gather(k, pair_idx), knn_gather(v, pair_idx)
        k, v = torch.max(k, dim=2)[0], torch.max(v, dim=2)[
            0]  # [B, global_points_num, attn_dim]
        k, v = knn_gather(k, inv_pair_idx), knn_gather(
            v, inv_pair_idx)  # [B, N, k_nearest, attn_dim]
        global_attn = self.global_fc(
            q[:, :, None, :] - k + pos_enc) / self.attn_scale
        global_attn = self.softmax(global_attn)
        global_attn = global_attn * (v + pos_enc)
        global_attn = torch.sum(global_attn, dim=2)  # [B, N, attn_dim]
        return global_attn

    def _get_self_idx(self, k_idx):
        b, n, k = k_idx.shape
        if not hasattr(self, 'idx') or self.idx.shape != k_idx.shape or self.idx.device != k_idx.device:
            self.idx = torch.arange(n, device=k_idx.device)[
                None, :, None].repeat(b, 1, k)
        return self.idx


class EventTransformerBlock(nn.Module):
    def __init__(self, dim, attn_dim, mlp_ratio, k_nearest, h, w, conv_kernel_size, global_step, drop_path):
        super().__init__()
        self.dim = dim
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EventAttention(dim=dim, attn_dim=attn_dim, k_nearest=k_nearest, h=h, w=w,
                                   conv_kernel_size=conv_kernel_size,
                                   global_step=global_step)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_ch=dim, hidden_ch=mlp_dim, out_ch=dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, xyzp, features):
        shortcut = features
        x = self.attn(xyzp, self.norm1(features))
        x = x + shortcut

        shortcut = x
        x = self.mlp(self.norm2(x))
        x = self.drop_path(x) + shortcut
        return x


class BasicLayer(nn.Module):
    def __init__(self, depth, down_stride, dim, attn_dim, mlp_ratio, k_nearest, h, w, conv_kernel_size, global_step,
                 drop_path):
        super().__init__()
        self.blocks = nn.ModuleList(
            EventTransformerBlock(
                dim=dim,
                attn_dim=attn_dim,
                mlp_ratio=mlp_ratio,
                k_nearest=k_nearest,
                h=h,
                w=w,
                conv_kernel_size=conv_kernel_size,
                global_step=global_step,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth))
        self.down = TransitionDown(in_ch=dim, out_ch=dim * 2, k_nearest=k_nearest,
                                   stride=down_stride) if down_stride > 0 else None

    def forward(self, xyzp, features):
        for blk in self.blocks:
            features = blk(xyzp, features)
        if self.down:
            xyzp, features = self.down(xyzp, features)
        return xyzp, features


class EventEmbed(nn.Module):
    def __init__(self, h, w, dim, norm_layer=None):
        super().__init__()
        self.sparse_embed = SparseConv(h, w, 0, 32, 9, mode='sum')

        self.pos_embed = nn.Linear(1, dim)
        self.neg_embed = nn.Linear(1, dim)

        self.proj = nn.Linear(64, 32, bias=False)
        self.norm = norm_layer(dim) if norm_layer else None

    def forward(self, x):
        x = x.clone().detach()
        sparse = self.sparse_embed(x, None)
        pos = x[..., 3][..., None]
        neg = 1 - x[..., 3][..., None]
        t = x[..., 2:3]
        x = pos * self.pos_embed(t) + neg * self.neg_embed(t)
        x = torch.cat([sparse, x], -1)
        x = self.proj(x)

        if self.norm:
            x = self.norm(x)
        return x


class EventTransformer(nn.Module):
    def __init__(self,
                 embed_dim=32,
                 embed_norm=True,
                 num_classes=10,
                 drop_path_rate=0.2,
                 height=128,
                 width=128,
                 conv_ks_list=[5, 3, 3, 3, 3],
                 depth_list=[1, 1, 1, 1, 1],
                 k_nearest_list=[16, 16, 16, 16, 16],
                 mlp_ratio_list=[4, 4, 4, 4, 4],
                 global_step_list=[16, 16, 8, 4, 1],
                 down_stride_list=[4, 4, 4, 4, -1]):  # Global Transformer args
        super().__init__()
        self.embed = EventEmbed(
            height, width, dim=embed_dim, norm_layer=nn.LayerNorm if embed_norm else None)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depth_list))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i in range(len(depth_list)):
            layer = BasicLayer(
                depth=depth_list[i],
                down_stride=down_stride_list[i],
                dim=int(embed_dim * 2 ** i),
                attn_dim=min(int(embed_dim * 2 ** i), 128),
                mlp_ratio=mlp_ratio_list[i],
                k_nearest=k_nearest_list[i],
                h=height // (2 ** i),
                w=width // (2 ** i),
                conv_kernel_size=conv_ks_list[i],
                global_step=global_step_list[i],
                drop_path=dpr[sum(depth_list[:i]):sum(depth_list[:i + 1])])
            self.layers.append(layer)

        num_features = int(embed_dim * 2 ** (len(depth_list) - 1))
        self.norm = nn.LayerNorm(num_features)
        self.head = nn.Linear(
            num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, xyzp):
        f = self.embed(xyzp)
        for i, layer in enumerate(self.layers):
            xyzp, f = layer(xyzp, f)

        f = self.norm(f)  # B, N, F
        f = f.mean(1)  # B, F
        return f

    def forward(self, events):
        f = self.forward_features(events)
        return self.head(f)
