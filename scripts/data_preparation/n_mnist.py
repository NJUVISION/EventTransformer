import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def bin_to_h5(h5_path, json_dir, bin_paths):
    h5f = h5py.File(h5_path, 'w')
    labels = {}
    labels['classes'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    h = w = 34
    for path in tqdm(bin_paths):
        cache = 500000
        timestamps = np.zeros([cache], dtype=np.uint32)
        pos = np.zeros([cache, 2], dtype=np.uint8)
        events = np.zeros([cache], dtype=np.bool)
        counter = 0
        with open(path, 'rb') as f:
            # read events
            while True:
                data = f.read(5)
                if not data:
                    break
                x = data[0]
                y = data[1]
                polarity = data[2] >> 7
                polarity = (polarity - 0.5) / 0.5
                timestamp = (data[2] & 127) << 16
                timestamp += data[3] << 8
                timestamp += data[4]

                timestamps[counter] = timestamp
                pos[counter, 0] = x
                pos[counter, 1] = y
                events[counter] = polarity
                counter += 1

        key = '_'.join(path.split('/')[-3:])
        h5f.create_dataset('{}_timestamps'.format(key), data=timestamps[:counter])
        dset = h5f.create_dataset('{}_pos'.format(key), data=pos[:counter])
        dset.attrs['size'] = json.dumps(dict(height=h, width=w))
        h5f.create_dataset('{}_events'.format(key), data=events[:counter])
        labels[key] = key.split('_')[1]

    classes = labels.pop('classes')
    train_keys = [k for k in labels.keys() if 'Train' in k]
    train, test = {'classes': classes}, {'classes': classes}
    for k in labels.keys():
        if k in train_keys:
            train[k] = labels[k]
        else:
            test[k] = labels[k]
    with open(os.path.join(json_dir, 'train.json'), 'w+') as f:
        json.dump(train, f)
    with open(os.path.join(json_dir, 'test.json'), 'w+') as f:
        json.dump(test, f)


if __name__ == '__main__':
    n_mnist_path = '/datasets/DVS/N-MNIST/'
    mat_paths = [str(s) for s in Path(n_mnist_path).glob("Train/*/*.bin")]
    mat_paths += [str(s) for s in Path(n_mnist_path).glob("Test/*/*.bin")]
    h5_path = os.path.join(n_mnist_path, 'n_mnist.h5')
    bin_to_h5(h5_path, n_mnist_path, mat_paths)
