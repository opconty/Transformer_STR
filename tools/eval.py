#-*- coding: utf-8 -*-
#'''
# @date: 2020/5/19 下午5:01
#
# @author: laygin
#
#'''
import torch
import torch.nn.functional as F
import time
from nltk.metrics.distance import edit_distance
import importlib

utils = importlib.import_module('utils')
Averager = utils.Averager


def validation(model, criterion, eval_loader, converter, device, cfg):
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(eval_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([cfg.batch_max_length] * batch_size).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=cfg.batch_max_length)

        start_time = time.time()

        with torch.no_grad():
            preds = model(image)
        forward_time = time.time() - start_time

        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), text_for_loss.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        labels = converter.decode(text_for_loss, length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score of one batch
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            gt = gt[:gt.find('<eos>')]
            pred_EOS = pred.find('<eos>')
            pred = pred[:pred_EOS]
            pred_max_prob = pred_max_prob[:pred_EOS]

            # fixme: do not care the case even case-sensitive model
            if not cfg.sensitive:
                pred = pred.lower()
                gt = gt.lower()

            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

