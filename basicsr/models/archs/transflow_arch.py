import torch
from pytorch3d.ops import knn_gather, ball_query
from torch import nn

from basicsr.models.archs.arch_util import TransitionUp, SwapAxes
from basicsr.models.archs.transcls_arch import EventTransformerBlock, EventTransformer


class BasicUpLayer(nn.Module):
    def __init__(self, depth, down_stride, dim, expand, attn_dim, mlp_ratio, k_nearest, h, w, conv_kernel_size,
                 global_step,
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
        self.up = TransitionUp(in_ch=dim * expand, out_ch=dim) if down_stride > 0 else None

    def forward(self, ori_xyzp, ori_features, down_xyzp=None, down_features=None):
        if self.up:
            assert ori_xyzp is not None and ori_features is not None
            ori_features = self.up(
                down_xyzp=down_xyzp,
                down_features=down_features,
                ori_xyzp=ori_xyzp,
                ori_features=ori_features)
        for blk in self.blocks:
            ori_features = blk(ori_xyzp, ori_features)
        return ori_features


class FlowEmbed(nn.Module):
    def __init__(self, dim, radius, h, w, k_nearest=32):
        super().__init__()
        self.radius = radius
        self.k_nearest = k_nearest
        self.res = torch.FloatTensor([w, h])[None, None, None, :]
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 9, dim),
            SwapAxes(1, 3),
            nn.BatchNorm2d(dim),
            SwapAxes(1, 3),
            nn.ReLU(inplace=True),
        )
        self._self_idx = None
        self._k_idx_shape = None
        self._device = None

    def forward(self, xyzp, features):
        xyz = xyzp[..., :3].clone().detach()
        xyz[..., 2] = 0
        idx = ball_query(xyz, xyz, radius=self.radius, K=self.k_nearest).idx  # [B, N. k_nearest]
        self._k_idx_shape = idx.shape
        self._device = idx.device
        idx = torch.where(idx == -1, self.self_idx, idx)  # replace -1 idx to self idx

        d_xyzp = xyzp[:, :, None, :] - knn_gather(xyzp, idx)  # [B, N, k_nearest, 4]
        ts = d_xyzp[..., 2][..., None]
        ts = torch.where(ts == 0, torch.ones_like(ts) * 1e-6, ts)
        speed = d_xyzp[..., :2] / ts * self.res.to(self._device)  # [B, N, k_nearest, 2]
        mean_speed = speed.mean(2, keepdim=True).mean(1, keepdim=True)  # [B, 1, k_nearest, 2]
        mean_speed = mean_speed.repeat([1, speed.shape[1], self.k_nearest, 1])  # [B, N, k_nearest, 2]
        p = torch.abs(d_xyzp[..., 3][..., None])

        knn_f = knn_gather(features, idx)  # [B, N, k_nearest, C]
        f = features[:, :, None, :].repeat([1, 1, self.k_nearest, 1])  # [B, N, k_nearest, C]
        f = torch.cat([f, knn_f, d_xyzp, speed, p, mean_speed], dim=-1)  # [B, N, k_nearest, C+C+4+2+1]
        f = self.mlp(f)
        f = torch.max(f, dim=-2)[0]  # [B, N, C]
        return f

    @property
    def self_idx(self):
        if self._self_idx is None \
                or self._self_idx.device != self._device \
                or self._self_idx.shape != self._k_idx_shape:
            b, n, k = self._k_idx_shape
            self._self_idx = torch.arange(n, device=self._device)[None, :, None].repeat(b, 1, k)
        return self._self_idx


class EventTransFlow(EventTransformer):
    def __init__(self,
                 embed_dim=32,
                 embed_norm=False,
                 drop_path_rate=0.,
                 height=128,
                 width=128,
                 depth_list=[1, 1, 1, 1, 1],
                 k_nearest_list=[16, 16, 16, 16, 16],
                 mlp_ratio_list=[4, 4, 4, 4, 4],
                 conv_ks_list=[5, 3, 3, 3, 3],
                 global_step_list=[32, 32, 16, 8, 2],
                 down_stride_list=[2, 4, 4, 4, -1]):  # Global Transformer args
        super().__init__(
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            drop_path_rate=drop_path_rate,
            height=height,
            width=width,
            conv_ks_list=conv_ks_list,
            depth_list=depth_list,
            k_nearest_list=k_nearest_list,
            mlp_ratio_list=mlp_ratio_list,
            global_step_list=global_step_list,
            down_stride_list=down_stride_list,
            num_classes=-1
        )
        self.enc = self.layers

        # build decoder
        self.dec = nn.ModuleList()
        for j in range(len(depth_list)):
            i = len(depth_list) - j - 1
            layer = BasicUpLayer(
                depth=depth_list[i],
                down_stride=down_stride_list[i],
                dim=int(embed_dim * 2 ** i),
                expand=2,
                attn_dim=min(int(embed_dim * 2 ** i), 128),
                mlp_ratio=mlp_ratio_list[i],
                k_nearest=k_nearest_list[i],
                h=height // (2 ** i),
                w=width // (2 ** i),
                conv_kernel_size=conv_ks_list[i],
                global_step=global_step_list[i],
                drop_path=[0 for _ in range(depth_list[i])])
            self.dec.append(layer)

        self.size = torch.FloatTensor([height, width])[None, None, :]
        self.flow_embed = FlowEmbed(h=height, w=width, dim=embed_dim * 4, radius=5 / height, k_nearest=16)

        self.preds = nn.ModuleList()
        for j in range(len(depth_list)):
            i = len(depth_list) - j - 1
            self.preds.append(nn.Linear(int(embed_dim * 2 ** i), 2))

    def encoder(self, xyzp):
        f = self.embed(xyzp)
        results = [[xyzp, f]]
        for i, layer in enumerate(self.enc):
            xyzp, f = layer(xyzp, f)
            if i == 1:
                f = self.flow_embed(xyzp, f)
            results.append([xyzp, f])
        return results

    def decoder(self, results):
        outputs = []
        down_xyzp, down_f = None, None
        for layer, (ori_xyzp, ori_f) in zip(self.dec, results[::-1]):
            f = layer(ori_xyzp, ori_f, down_xyzp, down_f)
            outputs.append((ori_xyzp, f))
            down_xyzp, down_f = ori_xyzp, f
        return outputs

    def predictor(self, outputs):
        preds = []
        for layer, (xyzp, f) in zip(self.preds, outputs):
            preds.append(layer(f))
        return preds

    def forward(self, events):
        self.device = events.device
        self.events_shape = events.shape

        results = self.encoder(events)
        results.pop(-2)

        outputs = self.decoder(results)
        preds = self.predictor(outputs)

        events_list = [xyzp for xyzp, f in outputs]
        flow_list = preds

        events_list = [events_list[-1]]
        flow_list = [flow_list[-1]]

        return events_list, flow_list
