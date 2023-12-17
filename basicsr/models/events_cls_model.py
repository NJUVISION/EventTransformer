import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from pytorch3d.ops import add_points_features_to_volume_densities_features
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class TransClsModel(BaseModel):
    """Base classification model for events classification."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.cal_flops_and_params(opt['point_num'])
        self.net_g = self.model_to_device(self.net_g)

        self.classes = self.opt.get('classes', None)
        self.use_bce = self.opt['network_g']['num_classes'] == 2

        self.load_pretrain_network()
        if self.is_train:
            self.correct, self.counter = 0., 0.
            self.init_training_settings()

    def feed_data(self, data):
        self.events = data['events'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def setup_loss(self):
        self.loss_fn = torch.nn.CrossEntropyLoss() if not self.use_bce else torch.nn.BCEWithLogitsLoss()

    def split(self):
        b, n, _ = self.events.size()
        assert b == 1
        crop_size = self.opt['val'].get('sample_num')
        parts = []
        for i in range(0, n - crop_size, crop_size):
            start_id = i if i + crop_size < n else n - crop_size
            part_events = self.events[:, start_id:start_id + crop_size]
            part_events[:, :, 2] = (part_events[:, :, 2] - part_events[:, 0, 2]) / (
                    part_events[:, -1, 2] - part_events[:, 0, 2])
            parts.append(part_events)

        if len(parts) == 0:
            part_events = self.events
            part_events[:, :, 2] = (part_events[:, :, 2] - part_events[:, 0, 2]) / (
                    part_events[:, -1, 2] - part_events[:, 0, 2])
            parts.append(part_events)
        self.origin_events = self.events
        self.events = torch.cat(parts, dim=0)

    def split_inverse(self):
        idx = torch.argmax(self.output, dim=1).long()  # B
        b = torch.arange(idx.shape[0]).long()  # B
        self.output = torch.zeros_like(self.output)
        self.output[b, idx] = 1
        self.output = torch.mean(self.output, 0, keepdim=True)

        events = self.origin_events
        events[:, :, 2] = (events[:, :, 2] - events[:, 0, 2]) / (events[:, -1, 2] - events[:, 0, 2])
        self.events = events

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.events)
        loss_dict = OrderedDict()
        l_total = self.loss_fn(self.output, self.gt) if not self.use_bce else self.loss_fn(self.output[:, 0],
                                                                                           self.gt.float())
        loss_dict['l_cls'] = l_total
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward(retain_graph=True)

        use_grad_clip = self.opt['train'].get('use_grad_clip', False)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 5)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        # calculate acc:
        if current_iter % 1000 == 0:
            self.correct, self.counter = 0., 0.
        if self.use_bce:
            self.output[:, 0] = 1 - torch.sigmoid(self.output[:, 0])
            self.output[:, 1] = 1 - self.output[:, 0]
        pred_choice = self.output.data.max(1)[1]
        self.correct += pred_choice.eq(self.gt.long().data).cpu().sum()
        self.counter += self.output.shape[0]
        self.log_dict['acc'] = self.correct / self.counter

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.events.size(0)
            m = self.opt['val'].get('max_minibatch', n)
            self.output = self.net_g.inference(self.events, minibatch=m)
        if self.use_bce:
            self.output[:, 0] = 1 - torch.sigmoid(self.output[:, 0])
            self.output[:, 1] = 1 - self.output[:, 0]
        self.net_g.train()

    def single_image_inference(self, events, save_path=None):
        self.feed_data(data={'events': events.unsqueeze(dim=0)})

        if self.opt['val'].get('split') is not None:
            self.split()

        self.test()

        if self.opt['val'].get('split') is not None:
            self.split_inverse()

        visuals = self.get_current_visuals(save_img=save_path is not None)
        if save_path is not None:
            events = tensor2img([visuals['events']])
            imwrite(events, save_path)
        class_id = np.argmax(visuals['output'].numpy(), axis=1)[0]
        print('save_path: {} \n class_id: {} class: {}'.format(save_path, class_id, self.classes[class_id]))

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image, use_pbar=True, print_log=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='events')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['key'][0]))[0]
            self.key = val_data['key']
            self.feed_data(val_data)
            if self.opt['val'].get('split') is not None:
                self.split()
            description = f'{self.events.shape[0]} events '
            self.test()

            if self.opt['val'].get('split') is not None:
                self.split_inverse()

            visuals = self.get_current_visuals(save_img=save_img)

            # tentative for out of GPU memory
            del self.events
            del self.output
            if 'gt' in visuals.keys():
                del self.gt
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                events = tensor2img([visuals['events']])
                imwrite(events, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['output'], visuals['gt'], **opt_)

            cnt += 1
            if use_pbar:
                pbar.update(1)
                for name, value in self.metric_results.items():
                    description += f'{name}: {value / cnt:.2f} '
                description += f'Test {img_name}'
                pbar.set_description(description)
        if use_pbar:
            pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            if print_log:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def get_current_visuals(self, save_img=False):
        out_dict = OrderedDict()
        out_dict['output'] = self.output.detach().cpu()  # [B, classes_num]
        if self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()  # [B]
        if save_img:
            events = (self.events.clone() - 0.5) / 0.5
            h, w = self.opt['val'].get('height', 300), self.opt['val'].get('width', 300)
            volume_densities = torch.zeros([events.shape[0], 1, 1, h, w]).to(events.device)
            volume_features = torch.zeros_like(volume_densities)
            volume_features, _ = add_points_features_to_volume_densities_features(events[:, :, :3],
                                                                                  events[:, :, 3][:, :, None],
                                                                                  volume_densities,
                                                                                  volume_features)  # [B, 1, 1, h, w]
            events = volume_features[:, None, 0, 0].detach().cpu()  # [B, 1, h, w]
            out_dict['events'] = (events + 1) / 2
        return out_dict
