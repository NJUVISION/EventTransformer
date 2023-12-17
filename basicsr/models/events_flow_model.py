import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import flow_vis_torch
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import imwrite, flow_to_color

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class TransFlowModel(BaseModel):
    """Base FLow model for events optical flow estimation."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.cal_flops_and_params(opt['point_num'])
        self.height, self.width = opt['height'], opt['width']
        self.net_g = self.model_to_device(self.net_g)

        self.loss_fn = loss_module.MaskedL2Loss()

        # load pretrained models
        self.load_pretrain_network()
        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        self.events = data['events'].to(self.device)
        self.events_cnt = data['events_cnt'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.gt_frame = data['gt_frame'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.loss_scale = data['loss_scale'].to(self.device)

    def setup_loss(self):
        self.loss_fn = loss_module.MaskedL2Loss()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # self.dt = self.events[:, -1:, 2:3] - self.events[:, :1, 2:3]
        self.dt = 5e-3
        events = self.events.clone()
        events[:, :, 2:3] = events[:, :, 2:3] / self.dt
        self.output = self.net_g(events)

        loss_dict = OrderedDict()
        _, flow_list = self.output
        l_total = self.loss_fn(flow_list[-1] / self.dt * self.loss_scale.view([-1, 1, 1]), self.gt)
        loss_dict['l2'] = l_total
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 5)

        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        with torch.no_grad():
            l_null = self.loss_fn(torch.zeros_like(flow_list[-1]), self.gt)
            self.log_dict['zero_flow_baselines'] = l_null

    def get_current_visual_log(self):
        events_list, flow_list = self.output
        with torch.no_grad():
            res = (self.height, self.width)
            pred = flow_to_color(events_list[-1], flow_list[-1] / self.dt * self.loss_scale.view([-1, 1, 1]), res)
            gt = flow_to_color(self.events, self.gt, res)
            imgs_dict = dict(
                pred=make_grid(pred.detach(), padding=10, pad_value=1),
                gt=make_grid(gt.detach(), padding=10, pad_value=1)
            )

        return imgs_dict

    def _test(self, events):
        self.net_g.eval()
        with torch.no_grad():
            n = events.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                _, flow_list = self.net_g(events[i:j])
                pred = flow_list[-1]
                outs.append(pred)
                i = j
        self.net_g.train()
        return torch.cat(outs, dim=0)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image, use_pbar=True, print_log=True):
        dataset_name = dataloader.dataset.opt['name']
        self.metric_results = dict(AEE=0., OUT=0.)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='events')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            # crop events
            assert self.events.size(0) == 1
            events, dts = [], []
            n = self.events.size(1)
            pn = self.opt['point_num']
            for i in [_ for _ in range(0, n // pn * pn, pn)] + [n - pn]:
                part_events = self.events[:, i:i + pn].clone()
                assert part_events.shape[1] == pn, f'{part_events.shape}, {pn}, {n}'
                dt = part_events[:, -1:, 2] - part_events[:, :1, 2]
                dt = torch.ones_like(dt) * 5e-3
                part_events[:, :, 2] = (part_events[:, :, 2] - part_events[:, :1, 2]) / dt
                events.append(part_events)
                dts.append(dt)
            events = torch.cat(events, dim=0)
            output = self._test(events) / torch.cat(dts, dim=0).view([-1, 1, 1])
            drop_i = pn - (n - n // pn * pn)
            events = torch.cat([events[:-1].view([1, -1, 4]), events[-1:, drop_i:]], dim=1)
            output = torch.cat([output[:-1].view(1, -1, 2), output[-1:, drop_i:]], dim=1)

            x, y = events[:, :, 0], events[:, :, 1]
            h, w = self.height, self.width
            _idx = y * (h - 1) * w + x * (w - 1)
            _idx = _idx[:, :, None].repeat(1, 1, 2).round().long()
            self.output = torch.zeros([1, h * w, 2], device=self.gt.device)
            self.output.scatter_add_(1, _idx, output)
            _cnt = torch.zeros_like(self.output)
            _cnt.scatter_add_(1, _idx, torch.ones_like(output))
            self.output = self.output / torch.clamp_(_cnt, min=1)
            self.output = self.output.view(1, h, w, 2).permute(0, 3, 1, 2) * self.loss_scale

            aee, out = metric_module.calculate_AEE(self.output, self.gt_frame, self.mask)

            self.metric_results['AEE'] += aee
            self.metric_results['OUT'] += out

            if save_img:
                # img_name = val_data['key']
                img_name = str(idx)
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                visual_img = self.get_current_visuals()
                imwrite(visual_img, save_img_path)

            cnt += 1
            if use_pbar:
                pbar.update(1)
                aee, out = self.metric_results['AEE'], self.metric_results['OUT']
                dt, num = self.events[0, -1, 2] - self.events[0, 0, 2], self.events.shape[1]
                pbar.set_description(
                    f'Test {dataset_name} {idx + 1}/{len(dataloader)} AEE: {aee / cnt:.4f} OUT: {out / cnt:.4f} dt: {dt:.5f} num: {num}')

            del self.events
            del self.output
            del self.gt
            # del self.mask
            torch.cuda.empty_cache()

        if use_pbar:
            pbar.close()

        for metric in self.metric_results.keys():
            self.metric_results[metric] /= cnt
        if print_log:
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        current_metric = self.metric_results['AEE']
        return current_metric

    def single_image_inference(self, events, save_path):
        raise NotImplementedError

    def get_current_visuals(self):
        pred_gt = torch.cat([self.output, self.gt_frame], dim=0)
        pred_gt = flow_vis_torch.flow_to_color(pred_gt / 5, normalize_over_video=False)
        pred_gt *= self.mask
        pred_gt = pred_gt.permute(0, 2, 3, 1).cpu().numpy()
        pred_gt = np.concatenate([pred_gt[0], pred_gt[1]], axis=1)
        pred_gt = np.clip(pred_gt, 0, 255).astype(np.uint8)
        return pred_gt
