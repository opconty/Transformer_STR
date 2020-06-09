#-*- coding: utf-8 -*-
#'''
# @date: 2020/5/18 下午6:06
#
# @author: laygin
#
#'''
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import math


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


def get_img_tensor(img_path, newh=32, neww=100, keep_ratio=False):
    if isinstance(img_path, str):
        image = Image.open(img_path).convert('L')
    elif isinstance(img_path, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)).convert('L')
    else:
        raise Exception(f'{type(img_path)} not supported yet')

    if keep_ratio:  # same concept with 'Rosetta' paper
        resized_max_w = neww
        input_channel = 3 if image.mode == 'RGB' else 1
        transform = NormalizePAD((input_channel, newh, resized_max_w))

        w, h = image.size
        ratio = w / float(h)
        if math.ceil(h * ratio) > neww:
            resized_w = neww
        else:
            resized_w = math.ceil(newh * ratio)

        resized_image = image.resize((resized_w, newh), Image.BICUBIC)
        t = transform(resized_image)

    else:
        transform = ResizeNormalize((neww, newh))
        t = transform(image)
    # print(f'image size: {image.size}\t tensor size: {t.size()}')
    return torch.unsqueeze(t, 0)


class TransLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, character, device):
        self.device = device
        list_token = ['<eos>']
        self.character = list_token + list(character)

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('<eos>')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return batch_text.to(self.device), torch.IntTensor(length).to(self.device)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
