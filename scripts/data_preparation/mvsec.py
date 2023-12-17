import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

MVSEC_ORI_PATH = '/datasets/MVSEC/hdf5'
MVSEC_FLOW = '/datasets/MVSEC/Flow_GT'
MVSEC_META_PATH = '/datasets/MVSEC/meta'
START_TIME = dict(
    indoor_flying1=4,
    indoor_flying2=9,
    indoor_flying3=7,
    indoor_flying4=6,
    outdoor_day1=3,
    outdoor_day2=45,
)
HOT_MAX_PX = 100
HOT_MAX_RATE = 0.8


def map_flow_to_events(ori_h5_path, flow_npz_path, meta_path, start_time):
    # load events
    # [e_num, 4]
    with h5py.File(ori_h5_path, 'r') as data:
        events = np.array(data['davis']['left']['events'][:])
        e_num = events.shape[0]

    # load flow
    # [f_num, H, W]
    with np.load(flow_npz_path) as data:
        ts = data['timestamps']
        y_flow, x_flow = data['y_flow_dist'], data['x_flow_dist']
        print(f'height: {y_flow.shape[1]}, width: {y_flow.shape[2]}')
        f_num = y_flow.shape[0]

    # pair events and flow
    e_idx, f_idx = np.searchsorted(events[:, 2], ts[0], side="right") - 1, 0
    map_f2e = []
    events_speed = np.zeros([e_num, 2], dtype=np.float32)
    hot_cnt = 0
    hot_mask = np.zeros_like(y_flow[0]).astype(np.float32)

    print(f'Jump events num: {e_idx}')
    start_idx = e_idx
    with tqdm(total=e_num - start_idx) as pbar:
        while e_idx < e_num - 1 and f_idx < f_num - 1:
            _e_idx = e_idx + \
                     np.searchsorted(events[e_idx:, 2], ts[f_idx + 1], side='right')
            if ts[f_idx] - ts[0] > start_time and _e_idx < e_num - 1:
                # map
                map_f2e.append([f_idx, e_idx, _e_idx])
                # speed
                pos_x = events[e_idx:_e_idx, 0].astype(np.uint32)
                pos_y = events[e_idx:_e_idx, 1].astype(np.uint32)
                events_speed[e_idx:_e_idx, 0] = y_flow[f_idx][pos_y,
                                                              pos_x] / (ts[f_idx + 1] - ts[f_idx])
                events_speed[e_idx:_e_idx, 1] = x_flow[f_idx][pos_y,
                                                              pos_x] / (ts[f_idx + 1] - ts[f_idx])
                # hot pixel
                events_cnt_img = np.zeros_like(hot_mask)
                events_cnt_img[pos_y, pos_x] += 1
                hot_mask += np.clip(events_cnt_img, 0, 1)
                hot_cnt += 1

            pbar.update(_e_idx - e_idx)
            pbar.set_description(
                f'{f_idx}/{f_num}, delta_e:{_e_idx - e_idx}, delta_t:{ts[f_idx + 1] - events[_e_idx - 1, 2]}, duration:{ts[f_idx + 1] - ts[f_idx]} ')
            f_idx += 1
            e_idx = _e_idx

    # save to h5
    map_f2e = np.array(map_f2e, dtype=np.int32)
    print(f'Saving events speed to {meta_path} num {map_f2e.shape[0]}')
    d = meta_path[:-len(os.path.basename(meta_path))]
    if not os.path.exists(d):
        os.makedirs(d)
    with h5py.File(meta_path, 'w') as data:
        data.create_dataset('flow_to_events_idx', data=map_f2e)
        data.create_dataset('events_speed', data=events_speed)

        # save hot pixel
        events_rate = hot_mask / hot_cnt
        hot_mask = np.ones_like(events_rate)
        for _ in range(HOT_MAX_PX):
            argmax = np.argmax(events_rate)
            index = (
                argmax // events_rate.shape[1], argmax % events_rate.shape[1])
            if events_rate[index] > HOT_MAX_RATE:
                events_rate[index] = 0
                hot_mask[index] = 0
            else:
                break
        data.create_dataset('hot_mask', data=hot_mask)

        data.close()


if __name__ == '__main__':
    ori_h5_path = os.path.join(MVSEC_ORI_PATH, '{}', '{}_data.hdf5')
    flow_npz_path = os.path.join(MVSEC_FLOW, '{}', '{}_gt_flow_dist.npz')
    meta_path = os.path.join(MVSEC_META_PATH, '{}', '{}_meta.hdf5')
    for p in [str(p) for p in Path(MVSEC_ORI_PATH).glob('*/*data.hdf5')]:
        d, n = p.split('/')[-2], os.path.basename(p)[:-len('_data.hdf5')]
        print(f'Processing dir {d} name {n}')
        map_flow_to_events(ori_h5_path.format(d, n), flow_npz_path.format(d, n), meta_path.format(d, n),
                           start_time=START_TIME[n])
