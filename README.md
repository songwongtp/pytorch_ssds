# pytorch_ssds_viewAnalysis
This repo is based on the work of [ssd.pytorch](https://github.com/ShuangXieIrene/ssds.pytorch)

## Installation
1. pytorch == 0.3.1
2. install requirements by `pip install -r ./requirements.txt`

## Usage
To train, test and demo some specific model. Please run the relative file in folder with the model configure file, like:

`python train.py --cfg=./experiments/cfgs/attfssd_lite_mobilenetv2_train_house_embed_att.yml`

`python test.py --cfg=./experiments/cfgs/attfssd_lite_mobilenetv2_eval_house_embed_att.yml`

Change the configure file based on the note in [config_parse.py](./lib/utils/config_parse.py)
