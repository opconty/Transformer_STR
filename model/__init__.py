#-*- coding: utf-8 -*-
#'''
# @date: 2020/6/9 下午12:17
#
# @author: laygin
#
#'''
import torch
import torch.nn as nn
import importlib
from config import logger
model_util = importlib.import_module('utils.model_util')

TPS_STN = model_util.TPS_STN
ResNet50 = model_util.ResNet50
BiLSTM = model_util.BiLSTM
Transformer = model_util.Transformer


class Model(nn.Module):
    def __init__(self,
                 imgh=32,
                 imgw=100,
                 input_channel=1,
                 output_channel=512,
                 hidden_size=256,
                 num_fiducial=20,
                 num_class=41,
                 bilstm=True,
                 device=torch.device('cuda:0')):
        super(Model, self).__init__()

        logger.info(f'bi-lstm: {bilstm} | device: {device} | num_class: {num_class}')
        self.num_class = num_class
        self.bilstm = bilstm

        self.transformation = TPS_STN(num_fiducial, I_size=(imgh, imgw), I_r_size=(imgh, imgw), device=device,
                                      I_channel_num=input_channel)
        self.fe = ResNet50(input_channel, output_channel)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.seq = nn.Sequential(BiLSTM(output_channel, hidden_size, hidden_size),
                                 BiLSTM(hidden_size, hidden_size, hidden_size))
        if self.bilstm:
            self.seq_out_channels = hidden_size
        else:
            logger.warn('There is no sequence model specified')
            self.seq_out_channels = output_channel
        self.prediction = Transformer(self.num_class, self.seq_out_channels)

    def forward(self, x):
        x = self.transformation(x)
        x = self.fe(x)
        x = self.adaptive_avg_pool(x.permute(0,3,1,2))  # [b, c, h, w] -> [b, w, c, h]
        x = x.squeeze(3)

        if self.bilstm:
            x = self.seq(x)

        pred = self.prediction(x.contiguous())
        return pred
