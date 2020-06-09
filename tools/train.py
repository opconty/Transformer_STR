#-*- coding: utf-8 -*-
#'''
# @date: 2020/6/9 下午12:26
#
# @author: laygin
#
#'''
import os, sys, time, random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import string
import config
import importlib

utils = importlib.import_module('utils')
eval = importlib.import_module('tools.eval')
data = importlib.import_module('data')
model = importlib.import_module('model')

TransLabelConverter = utils.TransLabelConverter
Averager = utils.Averager
validation = eval.validation
hierarchical_dataset = data.hierarchical_dataset
AlignCollate = data.AlignCollate
BatchBalancedDataset = data.BatchBalancedDataset
Model = model.Model


logger = config.logger
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Config(config.Config):
    train_data = os.path.join(config.data_dir, 'training')
    valid_data = os.path.join(config.data_dir, 'validation')
    saved_model = ''
    select_data = ['SJ', 'MJ']  # select training data
    batch_ratio = [0.5, 0.5]  # assign ratio for each selected data in the batch
    workers = 0
    batch_size = 192
    num_iter = 3000000
    valInterval = 2000
    total_data_usage_ratio = 1.0
    num_class = 40
    with_bilstm = True
    sensitive = False

    checkpoint_dir = config.checkpoint_dir
    Name = f'trans_{"".join(select_data)}_brnn{with_bilstm}'


def train(cfg):
    """ dataset preparation """
    if not cfg.data_filtering_off:
        logger.info('Filtering the images containing characters which are not in character')
        logger.info(f'Filtering the images whose label is longer than {cfg.batch_max_length}')

    if cfg.sensitive:
        cfg.character = string.digits + string.ascii_letters + cfg.punctuation
        cfg.Name += '_sensitive'

    train_dataset = BatchBalancedDataset(cfg.train_data, cfg.batch_max_length,cfg.character,
                                         cfg.select_data, cfg.batch_ratio, cfg.batch_size, cfg.total_data_usage_ratio,
                                         cfg.workers, cfg.sensitive, cfg.rgb, cfg.data_filtering_off, cfg.imgH, cfg.imgW,
                                         cfg.keep_ratio_with_pad)

    AlignCollate_valid = AlignCollate()
    valid_dataset = hierarchical_dataset(cfg.valid_data, cfg.imgH, cfg.imgW, cfg.batch_max_length, cfg.character,
                                         cfg.sensitive, cfg.rgb, cfg.data_filtering_off)
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=int(cfg.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    """model"""
    converter = TransLabelConverter(cfg.character, device)
    cfg.num_class = len(converter.character)
    logger.verbose(f'{cfg.num_class}\n{converter.character}')

    if cfg.rgb:
        cfg.input_channel = 3
    model = Model(cfg.imgH,cfg.imgW, cfg.input_channel, cfg.output_channel, cfg.hidden_size,
                  cfg.num_fiducial, cfg.num_class, cfg.with_bilstm,
                  device=device)

    logger.info('initialize')
    for name, param in model.named_parameters():
        if 'localization_fc2' in name or 'decoder' in name or 'self_attn' in name:
            logger.info(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
        except:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if os.path.isfile(cfg.saved_model):
        logger.info(f'loading pretrained model from {cfg.saved_model}')
        model.load_state_dict(torch.load(cfg.saved_model))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info('Trainable params num : ', sum(params_num))

    # setup optimizer
    optimizer = optim.Adadelta(filtered_parameters, lr=cfg.lr, rho=cfg.rho, eps=cfg.eps)

    """ start training """
    start_iter = 0
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    i = start_iter
    epoch_size = len(train_dataset) // cfg.batch_size

    while True:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        target, length = converter.encode(labels, batch_max_length=cfg.batch_max_length)

        preds = model(image)
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if i % cfg.valInterval == 0 and i > 0:
            elapsed_time = time.time() - start_time

            model.eval()
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                    model, criterion, valid_loader, converter, device, cfg)
            model.train()

            # training loss and validation loss
            loss_log = f'[{i}/{cfg.num_iter}({epoch_size})] Train loss: {loss_avg.val():0.5f},' \
                       f' Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            logger.info(loss_log)
            loss_avg.reset()

            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
            logger.info(current_model_log)

            # keep best accuracy model (on valid dataset)
            if current_accuracy > best_accuracy and current_accuracy > 0.65:
                best_accuracy = current_accuracy
                best_norm_ED = current_norm_ED if current_norm_ED < best_norm_ED else best_norm_ED
                acc = f'{best_accuracy:.4f}'.replace('.', '')
                # save
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f'{cfg.Name}_iter{i + 1}_acc{acc}.pth'))
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
                logger.success(best_model_log)

            # show some predicted results
            print('-' * 80)
            print(f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F')
            print('-' * 80)
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                gt = gt[:gt.find('<eos>')]
                pred = pred[:pred.find('<eos>')]

                print(f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred.lower() == gt.lower())}')
            print('-' * 80)

        # save model per 1e+5 iter.
        if (i + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), os.path.join(cfg.checkpoint_dir, f'{cfg.Name}_iter{i + 1}.pth'))

        if i == cfg.num_iter:
            print('end the training')
            sys.exit()
        i += 1


if __name__ == '__main__':
    cfg = Config()

    random.seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    train(cfg)



