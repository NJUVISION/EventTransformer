import json
import os
import struct
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def dat_to_h5(h5_path, json_dir, dat_paths):
    h5f = h5py.File(h5_path, 'w')
    train_labels = {}
    test_labels = {}
    train_labels['classes'] = ['car', 'background']
    test_labels['classes'] = ['car', 'background']
    for path in tqdm(dat_paths):
        cache = 500000
        timestamps = np.zeros([cache], dtype=np.uint32)
        pos = np.zeros([cache, 2], dtype=np.uint8)
        events = np.zeros([cache], dtype=np.bool)
        counter = 0
        with open(path, 'rb') as f:
            # skip header
            last_pos = 0
            line = f.readline()
            while line.startswith(b'%'):
                last_pos = f.tell()
                line = f.readline(10000)
            f.seek(last_pos, 0)
            # read ev type and size
            f.read(2)
            while True:
                data = f.read(8)
                if not data:
                    break
                timestamp32 = struct.unpack("<I", data[0:4])[0]
                address = struct.unpack("<I", data[4:8])[0]

                polarity = (address & 0x10000000) >> 28
                y = (address & 0xFFFC000) >> 14
                x = (address & 0x3FFF)

                timestamps[counter] = timestamp32
                pos[counter, 0] = x
                pos[counter, 1] = y
                events[counter] = polarity
                counter += 1
        label = 'background' if 'background' in path else 'car'
        set = 'train' if 'train' in path else 'test'
        key = '{}_{}_{}'.format(set, label, os.path.basename(path))
        h5f.create_dataset('{}_timestamps'.format(key), data=timestamps[:counter])
        dset = h5f.create_dataset('{}_pos'.format(key), data=pos[:counter])
        h, w = int(pos[:, 1].max()) + 1, int(pos[:, 0].max()) + 1
        dset.attrs['size'] = json.dumps(dict(height=h, width=w))
        h5f.create_dataset('{}_events'.format(key), data=events[:counter])
        if set == 'train':
            train_labels[key] = label
        else:
            test_labels[key] = label
    h5f.close()

    with open(os.path.join(json_dir, 'train.json'), 'w+') as f:
        json.dump(train_labels, f)
    with open(os.path.join(json_dir, 'test.json'), 'w+') as f:
        json.dump(test_labels, f)


if __name__ == '__main__':
    ncars_dvs_path = '/datasets/DVS/N-CARS/Prophesee_Dataset_n_cars/'
    dat_paths = [str(s) for s in Path(ncars_dvs_path).glob("*/*/*.dat")]
    h5_path = os.path.join(ncars_dvs_path, 'n_cars.h5')
    dat_to_h5(h5_path, ncars_dvs_path, dat_paths)
