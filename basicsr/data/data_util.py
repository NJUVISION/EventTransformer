import random

import numpy as np


def _events_rotation(events):
    x = events[:, 0].copy()
    y = events[:, 1].copy()
    events[:, 0] = y
    events[:, 1] = 1 - x
    return events


def events_sampling(data_len, sample_num, sample_fn):
    assert sample_fn in ['crop', 'all']
    if sample_fn == 'crop':
        if data_len > sample_num:
            start_id = np.random.randint(0, data_len - sample_num)
            stop_id = start_id + sample_num
        else:
            start_id, stop_id = 0, data_len
    else:
        start_id, stop_id = 0, data_len
    return start_id, stop_id


def events_padding(events, target_size):
    while events.shape[0] < target_size:
        sample_size = min(target_size - events.shape[0], events.shape[0])
        start_id = events.shape[0] - sample_size
        # reflect events
        _events = events[start_id:][::-1].copy()
        # reflect time
        _events[:, 2] = events[-1, 2] - _events[:, 2] + events[-1, 2]
        # reflect polarity
        _events[:, 3] = 1 - _events[:, 3]
        events = np.concatenate([events, _events], axis=0)
    return events.copy()


def events_augmentation(events):
    # flip t
    # if random.random() < 0.5:
    #     events = events[::-1]
    #     events[:, 2] = events[0, 2] - events[:, 2]
    #     events[:, 3] = 1 - events[:, 3]
    # flip x
    if random.random() < 0.5:
        events[:, 0] = 1 - events[:, 0]
    # rotation
    # if random.random() < 0.5:
    #    for _ in range(random.randint(1, 4)):
    #        events = _events_rotation(events)
    return events.copy()
