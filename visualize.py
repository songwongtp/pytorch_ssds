from __future__ import print_function

import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from lib import subimg_utils

import json

config_file = "./experiments/cfgs/attfssd_lite_mobilenetv2_eval_house_embed_att.yml"
HOUSE_CLASSES = ['__background__', # always index 0
        'door', 'garage_door', 'window']

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--path', dest='image_path',
            help='optional image path', default=None, type=str)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self):
        self.cfg = cfg
        self.preproc = preproc(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, -2)

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            # if torch.cuda.device_count() > 1:
                # self.model = torch.nn.DataParallel(self.model).module

        # Print the model architecture and parameters
        print('Model architectures:\n{}\n'.format(self.model))

        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        print('number of parameters: {}'.format(num_parameters))

        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # Set the logger
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX


    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}

        checkpoint = self.model.state_dict()
        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def test_model(self, image):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if epoch in self.cfg.TEST.TEST_SCOPE:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=max(self.cfg.TEST.TEST_SCOPE)))
                    self.resume_checkpoint(resume_checkpoint)
                    return self.visualize_image(self.model, image, self.detector, epoch,  self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            return self.visualize_image(self.model, image, self.detector, 0,  self.use_gpu)

    def visualize_image(self, model, image, detector, epoch, use_gpu):
        model.eval()

        num_classes = len(HOUSE_CLASSES)
        
        _t = Timer()
        
        height, width = image.shape[:2]

        final_boxes = []
        final_classes = []
        final_scores = []

        #"""
        ratio = 0.7
        cropped_width = int(math.ceil(width * ratio))
        cropped_height = int(math.ceil(height * ratio))
            
        if width >= height:
            width_step = [int(math.ceil(i * (1-ratio) / 3 * width)) for i in range(4)]
            height_step = [int(math.ceil(i * (1-ratio) / 2 * height)) for i in range(3)]
        else:
            width_step = [int(math.ceil(i * (1-ratio) / 2 * width)) for i in range(3)]
            height_step = [int(math.ceil(i * (1-ratio) / 3 * height)) for i in range(4)]
        
        step = 0
            
        _t.tic()
        for left in width_step:
            for upper in height_step:
                step += 1
                    
                w = min(cropped_width, width - left)
                h = min(cropped_height, height - upper)
                pos = [left, upper, left, upper]
                scale = [w, h, w, h]

                cropped_image = image[upper:upper+h, left:left+w]

                if use_gpu:
                    images = Variable(self.preproc(cropped_image)[0].unsqueeze(0).cuda(), volatile=True)
                else:
                    images = Variable(self.preproc(cropped_image)[0].unsqueeze(0), volatile=True)
    
                # forward
                out = model(images, phase='eval')
        
                # detect
                detections = detector.forward(out)

                # TODO: make it smart:
                for j in range(1, num_classes):
                    for det in detections[0][j]:
                        if det[0] > 0:
                            d = det.cpu().numpy()
                            #print (d)
                            final_boxes.append(d[1:] * scale + pos)
                            final_classes.append(j)
                            final_scores.append(d[0])
                t_boxes = np.array(final_boxes)
                t_classes = np.array(final_classes)
                t_scores = np.array(final_scores)

        """
        _t.tic()
        if use_gpu:
            images = Variable(self.preproc(raw_image)[0].unsqueeze(0).cuda(), volatile=True)
        else:
            images = Variable(self.preproc(raw_image)[0].unsqueeze(0), volatile=True)
        # forward
        out = model(images, phase='eval')
        # detect
        detections = detector.forward(out)

        scale = [width, height, width, height]
        for j in range(1, num_classes):
            for det in detections[0][j]:
                if det[0] > 0:
                    d = det.cpu().numpy()
                    #print (d)
                    final_boxes.append(d[1:] * scale)
                    final_classes.append(j)
                    final_scores.append(d[0])
        """

        time = _t.toc()
        print ("Time:", time)
        final_boxes = np.array(final_boxes)
        final_classes = np.array(final_classes)
        final_scores = np.array(final_scores)

        output = []
        if final_boxes.shape[0] == 0:
            print ("no predictions")
            subimg_utils.display_instances(image, np.array([]), np.array([]), HOUSE_CLASSES, np.array([]))
        else:
            pick = subimg_utils.non_max_suppression(final_boxes, final_scores, 0)
            subimg_utils.display_instances(image, final_boxes[pick], final_classes[pick], HOUSE_CLASSES, final_scores[pick])
            for idx in pick:
                output.append({'box': list(final_boxes[idx]), 'type': HOUSE_CLASSES[final_classes[idx]]})

        json_object = json.dumps(output)
        return json_object

def visualize(image):
    cfg_from_file(config_file)
    s = Solver()
    return s.test_model(image)

if __name__ == '__main__':
    args = parse_args()
    print (visualize(cv2.imread(args.image_path, cv2.IMREAD_COLOR)))
