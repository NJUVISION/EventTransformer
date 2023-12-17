import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from high_level import labels_to_json


def mat_to_h5(h5_path, json_dir, mat_paths):
    f = h5py.File(h5_path, 'w')
    labels = {}
    labels['classes'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                         'h', 'i', 'k', 'l', 'm', 'n',
                         'o', 'p', 'q', 'r', 's', 't',
                         'u', 'v', 'w', 'x', 'y']
    for path in tqdm(mat_paths):
        data = loadmat(path)
        timestamps = data['ts'][:, 0].astype(np.uint32)
        pos = np.concatenate([data['x'], data['y']], axis=1)
        events = data['pol'][:, 0].astype(np.bool)
        key = os.path.basename(path)
        f.create_dataset('{}_timestamps'.format(key), data=timestamps)
        dset = f.create_dataset('{}_pos'.format(key), data=pos)
        dset.attrs['size'] = json.dumps(dict(height=180, width=240))
        f.create_dataset('{}_events'.format(key), data=events)
        labels[key] = key.split('_')[0]
    f.close()

    labels_to_json(labels, json_dir)


if __name__ == '__main__':
    asl_dvs_path = '/datasets/DVS/ASL-DVS/'
    mat_paths = [str(s) for s in Path(asl_dvs_path).glob("*/*.mat")]
    h5_path = os.path.join(asl_dvs_path, 'asl_dvs.h5')
    mat_to_h5(h5_path, asl_dvs_path, mat_paths)
