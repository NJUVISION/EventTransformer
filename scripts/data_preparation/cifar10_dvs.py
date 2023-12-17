import json
import os
import struct
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from high_level import labels_to_json


def aedat2_to_h5(h5_path, json_dir, aedat2_paths):
    h5f = h5py.File(h5_path, 'w')
    labels = {}
    labels['classes'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'ship', 'truck', 'frog', 'horse']
    h = w = 128
    for path in tqdm(aedat2_paths):
        cache = 500000
        timestamps = np.zeros([cache], dtype=np.uint32)
        pos = np.zeros([cache, 2], dtype=np.uint8)
        events = np.zeros([cache], dtype=np.bool)
        counter = 0
        with open(path, 'rb') as f:
            # skip header
            last_pos = 0
            line = f.readline()
            while line.startswith(b'#'):
                last_pos = f.tell()
                line = f.readline(10000)
            f.seek(last_pos, 0)
            # read events
            tsoverflow = 0
            last_timestamp32 = -1
            while True:
                data = f.read(8)
                if not data:
                    break
                address = struct.unpack('>I', data[0:4])[0]
                timestamp32 = struct.unpack('>I', data[4:8])[0]

                polarity = (address & 0x1)
                # int to uint
                polarity = 1 - polarity
                y = (address & 0x7f00) >> 8
                x = (address & 0xfE) >> 1
                # flip x
                x = w - 1 - x
                if timestamp32 < last_timestamp32:
                    tsoverflow += 1
                last_timestamp32 = timestamp32
                timestamp = timestamp32 | tsoverflow << 31

                timestamps[counter] = timestamp
                pos[counter, 0] = x
                pos[counter, 1] = y
                events[counter] = polarity
                counter += 1

        key = os.path.basename(path)
        h5f.create_dataset('{}_timestamps'.format(key), data=timestamps[:counter])
        dset = h5f.create_dataset('{}_pos'.format(key), data=pos[:counter])
        dset.attrs['size'] = json.dumps(dict(height=h, width=w))
        h5f.create_dataset('{}_events'.format(key), data=events[:counter])
        labels[key] = key.split('_')[1]
    h5f.close()
    labels_to_json(labels, json_dir)


if __name__ == '__main__':
    cifar10_path = '/datasets/DVS/CIFAR10-DVS/'
    aedat2_paths = [str(s) for s in Path(cifar10_path).glob("*/*.aedat")]
    h5_path = os.path.join(cifar10_path, 'cifar10_dvs.h5')
    aedat2_to_h5(h5_path, cifar10_path, aedat2_paths)
