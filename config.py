#-*- coding: utf-8 -*-
#'''
# @date: 2020/6/9 下午12:17
#
# @author: laygin
#
#'''
import os
import sys
import re
import string
from nicelogger import ColorLogger


logger = ColorLogger()
proj_dir = os.path.abspath(os.path.dirname(__file__))
if proj_dir not in sys.path:
    sys.path.insert(0, proj_dir)


checkpoint_dir = os.path.join(proj_dir, 'checkpoints')
# fixme: data path configuration
data_dir = 'path/to/data_lmdb_text_recognition'

p = re.compile(r'[!"#$%&()*+,/:;<=>?@\\^_`{|}~]')


class Config:
    workers = 8
    batch_max_length = 25
    batch_size = 32
    imgH = 32
    imgW = 100
    keep_ratio = False
    rgb = False
    sensitive = False
    data_filtering_off = False
    keep_ratio_with_pad = False
    total_data_usage_ratio = 1.0

    punctuation = r"""'.-"""
    character = string.digits + string.ascii_lowercase + punctuation

    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    num_gpu = 1

    lr = 1.0
    grad_clip = 5
    beta1 = 0.9
    rho = 0.95
    eps = 1e-8
    manualSeed = 2020
