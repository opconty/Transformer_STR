#-*- coding: utf-8 -*-
#'''
# @date: 2020/6/9 下午12:17
#
# @author: laygin
#
#'''
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
import string
import config
import warnings
warnings.filterwarnings('ignore')
import importlib

utils = importlib.import_module('utils')
eval = importlib.import_module('tools.eval')
data = importlib.import_module('data')
model = importlib.import_module('model')

TransLabelConverter = utils.TransLabelConverter
hierarchical_dataset = data.hierarchical_dataset
AlignCollate = data.AlignCollate
Model = model.Model


logger = config.logger
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
p = re.compile(r'[!"#$%&()*+,/:;<=>?@\\^_`{|}~]')


class Config(config.Config):
    valid_data = os.path.join(config.data_dir, 'validation')
    workers = 0
    batch_size = 32
    num_class = 40
    with_bilstm = True
    sensitive = False
    filter_punctuation = False
    backbone = 'resnet'

    checkpoint_dir = config.checkpoint_dir
    saved_model = ''


def create_model(cfg: Config):
    """model"""
    cfg.sensitive = True if 'sensitive' in cfg.saved_model else False

    if cfg.sensitive:
        cfg.character = string.digits + string.ascii_letters + cfg.punctuation

    converter = TransLabelConverter(cfg.character, device)
    cfg.num_class = len(converter.character)

    if cfg.rgb:
        cfg.input_channel = 3
    model = Model(cfg.imgH, cfg.imgW, cfg.input_channel, cfg.output_channel, cfg.hidden_size,
                  cfg.num_fiducial, cfg.num_class, cfg.with_bilstm, device=device)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    assert os.path.exists(cfg.saved_model), FileNotFoundError(f'{cfg.saved_model}')

    if os.path.isfile(cfg.saved_model):
        logger.info(f'loading pretrained model from {os.path.relpath(cfg.saved_model, os.path.dirname(__file__))}')
        model.load_state_dict(torch.load(cfg.saved_model, map_location=device))

    model.eval()

    return model, converter


def validation(cfg: Config,  model, converter):
    cfg.sensitive = True if 'sensitive' in cfg.saved_model else False
    AlignCollate_valid = AlignCollate()
    valid_dataset = hierarchical_dataset(cfg.valid_data, cfg.imgH, cfg.imgW, cfg.batch_max_length, cfg.character,
                                         cfg.sensitive, cfg.rgb, cfg.data_filtering_off)
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(cfg.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    model.eval()

    n_correct = 0
    length_of_data = 0

    for i, (image_tensors, labels) in enumerate(valid_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([cfg.batch_max_length] * batch_size).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=cfg.batch_max_length)

        with torch.no_grad():
            preds = model(image)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        labels = converter.decode(text_for_loss, length_for_loss)

        # calculate accuracy & confidence score of one batch
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            gt = gt[:gt.find('<eos>')]
            pred_EOS = pred.find('<eos>')
            pred = pred[:pred_EOS]
            pred_max_prob = pred_max_prob[:pred_EOS]

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0.0

            if not cfg.sensitive:
                pred = pred.lower()

            # fixme: filter punctuation
            if cfg.filter_punctuation:
                pred = re.sub(p, '', pred)
                gt = re.sub(p, '', gt)

            if pred == gt:
                n_correct += 1

    accuracy = n_correct / float(length_of_data)

    return accuracy


def eval_cute80(cute80_data_dir):
    cfg = Config()
    cfg.saved_model = os.path.join(cfg.checkpoint_dir,
                                   'Transformer_STR_CUTE80_pretrained.pth')

    model, converter = create_model(cfg)

    cfg.valid_data = cute80_data_dir
    acc = validation(cfg, model, converter)

    logger.success(f'{acc:.6f}')


if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = True

    cute80_dir = os.path.join(config.data_dir, 'evaluation', 'CUTE80')
    eval_cute80(cute80_dir)

