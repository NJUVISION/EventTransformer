import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from high_level import labels_to_json


def bin_to_h5(h5_path, json_dir, bin_paths):
    h5f = h5py.File(h5_path, 'w')
    labels = {}
    classes = []
    for p in bin_paths:
        if p.split('/')[-2] not in classes:
            classes.append(p.split('/')[-2])
    assert len(classes) == 101
    labels['classes'] = classes
    for path in tqdm(bin_paths):
        with open(path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        timestamps = all_ts[td_indices]
        pos = np.concatenate([all_x[td_indices][:, None], all_y[td_indices][:, None]], axis=1).astype(np.uint8)
        events = all_p[td_indices].astype(np.bool)

        h, w = int(pos[:, 1].max()) + 1, int(pos[:, 0].max()) + 1
        key = '_'.join(path.split('/')[-2:])
        h5f.create_dataset('{}_timestamps'.format(key), data=timestamps)
        dset = h5f.create_dataset('{}_pos'.format(key), data=pos)
        dset.attrs['size'] = json.dumps(dict(height=h, width=w))
        h5f.create_dataset('{}_events'.format(key), data=events)
        labels[key] = path.split('/')[-2]
    f.close()

    labels_to_json(labels, json_dir)


if __name__ == '__main__':
    caltect_path = '/datasets/DVS/N-Caltech101/Caltech101/'
    bin_paths = [str(s) for s in Path(caltect_path).glob("*/*.bin")]
    h5_path = os.path.join(caltect_path, 'caltech101.h5')
    bin_to_h5(h5_path, caltect_path, bin_paths)
