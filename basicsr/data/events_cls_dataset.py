import json

import h5py
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.data_util import events_augmentation, events_sampling, events_padding


class EventsClsDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self._h5, self.h5_path = None, opt['h5']
        with open(opt['labels'], 'r') as f:
            self.labels = json.load(f)
            self.classes = self.labels.pop('classes')
            self.labels = [{'key': key, 'label_id': self.classes.index(self.labels[key])} for key in self.labels]

        self.sample_fn, self.sample_num = opt['sample_fn'], opt['sample_num']
        assert self.sample_fn in ['all', 'crop']
        assert self.sample_num > 0
        # sample_fn: all -> return all points; crop -> return subset point;
        self.augmentation = self.opt.get('augmentation', False)

    @property
    def h5(self):
        if self._h5 is None:  # lazy loading here!
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _get_events(self, key):
        data_len, size = self.h5[key + '_events'].shape[0], json.loads(self.h5[key + '_pos'].attrs['size'])
        size = np.array([[size['width'] - 1, size['height'] - 1]], dtype=np.float32)

        i, j = events_sampling(data_len=data_len, sample_fn=self.sample_fn, sample_num=self.sample_num)
        events = np.zeros([j - i, 4], dtype=np.float32)
        events[:, :2] = self.h5[key + '_pos'][i:j].astype(np.float32) / size
        events[:, 2] = self.h5[key + '_timestamps'][i:j].astype(np.float32)
        events[:, 3] = self.h5[key + '_events'][i:j]

        # repeat padding
        events = events_padding(events, self.sample_num)

        # only normalize when cropping events
        if self.sample_fn == 'crop':
            events[:, 2] = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2])

        if self.augmentation:
            events = events_augmentation(events)
        events = torch.from_numpy(events)
        return events

    def __getitem__(self, index):
        key = self.labels[index]['key']
        events = self._get_events(key)
        label_id = self.labels[index]['label_id']
        return {'events': events, 'gt': label_id, 'key': key}

    def __len__(self):
        return len(self.labels)
