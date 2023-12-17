import flow_vis_torch
import torch


def flow_to_color(events, flow, res, norm_batch=False):
    b, n, _ = flow.shape
    idx = events[..., (1, 0)]
    idx[..., 0] *= res[0] - 1  # y
    idx[..., 1] *= res[1] - 1  # x

    idx[..., 0] *= res[1]
    idx = torch.sum(idx, dim=-1, keepdim=True).long()
    idx = idx.repeat([1, 1, 2])

    flow_img = torch.zeros((b, res[0] * res[1], 2), device=idx.device)
    flow_img = flow_img.scatter_add_(1, idx.long(), flow)
    flow_mask = torch.where(flow_img != 0, torch.ones_like(flow_img), torch.zeros_like(flow_img))
    flow_mask = flow_mask[..., 0].view([b, 1, res[0], res[1]]).repeat([1, 3, 1, 1])

    flow_img = flow_img.view((b, res[0], res[1], 2)).permute([0, 3, 1, 2])
    flow_rgb = flow_vis_torch.flow_to_color(flow_img, normalize_over_video=norm_batch)
    return flow_rgb * flow_mask / 255
