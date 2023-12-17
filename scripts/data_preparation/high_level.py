import json
import os
import random


def labels_to_json(data, json_dir, train_ratio=0.8):
    classes = data.pop('classes')
    train_keys = random.sample(data.keys(), int(len(data.keys()) * train_ratio))
    train, test = {'classes': classes}, {'classes': classes}
    for k in data.keys():
        if k in train_keys:
            train[k] = data[k]
        else:
            test[k] = data[k]
    with open(os.path.join(json_dir, 'train.json'), 'w+') as f:
        json.dump(train, f)
    with open(os.path.join(json_dir, 'test.json'), 'w+') as f:
        json.dump(test, f)
