from random import random, randint

import h5py
import numpy as np
import torch
from torch.utils import data as data


class MVSECDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self._events_h5, self.events_h5_path = None, opt['events_h5']
        self._meta_h5, self.meta_h5_path = None, opt['meta_h5']
        self._idx_map = None

        height, width = opt['height'], opt['width']
        self.size = np.array((width - 1, height - 1), dtype=np.float32)

        self.mode, self.sample_num = opt['mode'], opt['sample_num']
        self.augmentation = opt.get('augmentation', False)
        self.dt = opt.get('dt', 1)
        assert self.mode in ['train', 'val']
        assert self.sample_num > 0
        assert self.dt in [1, 4]

    @property
    def events_h5(self):
        if self._events_h5 is None:  # lazy loading here!
            self._events_h5 = h5py.File(self.events_h5_path, 'r')
        return self._events_h5

    @property
    def meta_h5(self):
        if self._meta_h5 is None:
            self._meta_h5 = h5py.File(self.meta_h5_path, 'r')
        return self._meta_h5

    @property
    def idx_map(self):
        if self._idx_map is None:
            _idx_map = []
            for i in range(0, self.meta_h5['flow_to_events_idx'].shape[0] - self.dt, self.dt):
                if self.meta_h5['flow_to_events_idx'][i + self.dt - 1][2] \
                        - self.meta_h5['flow_to_events_idx'][i][1] \
                        > self.sample_num:
                    _idx_map.append(i)
            self._idx_map = _idx_map
            if 'outdoor_day1' in self.events_h5_path:
                self._idx_map = self._idx_map[-1700 // self.dt:-900 // self.dt]  # same as EVFlowNet
        return self._idx_map

    def _sparse_to_dense(self, events, sparse, avg=False):
        # events tensor [N, 4], tensor sparse [N, f]
        x = events[:, 0] * self.size[0]
        y = events[:, 1] * self.size[1]
        idx = y * (self.size[0] + 1) + x
        f = sparse.shape[-1]
        h, w = int(self.size[1]) + 1, int(self.size[0]) + 1

        idx = idx[None, :, None].repeat(1, 1, f).round().long()
        dense = torch.zeros([1, h * w, f])
        sparse = sparse[None, :, :]
        dense.scatter_add_(1, idx, sparse)
        if avg:
            cnt = torch.zeros([1, h * w, f])
            cnt.scatter_add_(1, idx, torch.ones_like(sparse))
            dense = dense / torch.clamp_(cnt, min=1)
        dense = dense.view(h, w, f).permute(2, 0, 1)
        return dense

    def _events_to_cnt(self, events):
        # events tensor [N, 4]
        pos_cnt = self._sparse_to_dense(events, events[:, 3:4], avg=False)
        neg_cnt = self._sparse_to_dense(events, 1 - events[:, 3:4], avg=False)
        cnt = torch.cat([pos_cnt, neg_cnt], dim=0)
        return cnt

    def _get_mask(self, events, flow):
        # generate mask
        # hot mask
        hot_mask = torch.from_numpy(np.array(self.meta_h5['hot_mask']).astype(np.float32))
        # flow valid mask
        flow_mask = torch.where(flow[0] == 0, torch.zeros_like(flow[0]), torch.ones_like(flow[0]))
        flow_mask *= torch.where(flow[1] == 0, torch.zeros_like(flow[1]), torch.ones_like(flow[1]))
        # events valid mask
        ys, xs = events[:, 1] * self.size[1], events[:, 0] * self.size[0]
        events_mask = torch.zeros_like(flow_mask)
        events_mask[ys.round().long(), xs.round().long()] = 1
        # fov mask
        fov_mask = torch.zeros_like(flow_mask)
        i, j = (self.size[0] + 1 - 256) // 2, (self.size[1] + 1 - 256) // 2
        i, j = int(i), int(j)
        fov_mask[j:j + 256, i:i + 256] = 1
        # car mask
        if 'outdoor' in self.events_h5_path:
            fov_mask[190:] = 0
        # combine mask
        mask = hot_mask * flow_mask * events_mask * fov_mask
        mask = mask[None, :, :]
        return mask

    def _augmentation(self, data):
        # Horizontal flip
        if random() < 0.5:
            data['events'][:, 0] = 1 - data['events'][:, 0]
            data['events_cnt'] = torch.flip(data['events_cnt'], dims=[2])
            data['gt'][:, 1] = -data['gt'][:, 1]
            data['gt_frame'] = torch.flip(data['gt_frame'], dims=[2])
            data['gt_frame'][1] = -data['gt_frame'][1]
            data['mask'] = torch.flip(data['mask'], dims=[2])
        # Vertical flip
        if random() < 0.5:
            data['events'][:, 1] = 1 - data['events'][:, 1]
            data['events_cnt'] = torch.flip(data['events_cnt'], dims=[1])
            data['gt'][:, 0] = -data['gt'][:, 0]
            data['gt_frame'] = torch.flip(data['gt_frame'], dims=[1])
            data['gt_frame'][0] = -data['gt_frame'][0]
            data['mask'] = torch.flip(data['mask'], dims=[1])
        return data

    def _get_data(self, idx):
        i, j = self.meta_h5['flow_to_events_idx'][idx][1], self.meta_h5['flow_to_events_idx'][idx + self.dt - 1][2]
        if self.mode == 'train':
            i = randint(i, j - self.sample_num)
            j = i + self.sample_num
        events = self.events_h5['davis']['left']['events'][i:j]
        events[:, :2] /= self.size
        events[:, 2] = events[:, 2] - events[0, 2]
        events[:, 3] = (events[:, 3] + 1) / 2
        events = events.astype(np.float32)
        events = torch.from_numpy(events)

        loss_scale = torch.FloatTensor([22e-3 * self.dt])

        gt = self.meta_h5['events_speed'][i:j].astype(np.float32)
        gt = torch.from_numpy(gt) * loss_scale

        events_cnt = self._events_to_cnt(events)
        gt_frame = self._sparse_to_dense(events, gt, avg=True)
        mask = self._get_mask(events, gt_frame)

        data = dict(events=events, events_cnt=events_cnt, gt=gt, gt_frame=gt_frame, mask=mask, loss_scale=loss_scale)
        return data

    def __getitem__(self, index):
        index = self.idx_map[index]
        data = self._get_data(index)
        if self.mode == 'train' and self.augmentation:
            data = self._augmentation(data)
        return data

    def __len__(self):
        return len(self.idx_map)
