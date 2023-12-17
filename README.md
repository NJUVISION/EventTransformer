# Event Transformer

This is the source code of Token-based Spatiotemporal Representation of the Events.

## Environment
Python ≥ 3.7 \
CUDA 11.8

## Installation
```
pip install -r requirements.txt
python setup.py develop
```
Require additional install [sparseconvnet](https://github.com/facebookresearch/SparseConvNet), [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and [flow_vis_torch](https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch)

## Datasets preparation
See [datasets.md](./datasets/datasets.md)

## Train
```
python basicsr/train.py -opt options/train/[DATASETS]/XXX.yml
```

## Test 
```
python basicsr/test.py -opt options/train/[DATASETS]/XXX.yml --pretrained experiments/[YML_NAME]/models/net_g_latest.pth
```

