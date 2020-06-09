#-*- coding: utf-8 -*-
#'''
# @date: 2020/5/19 上午10:21
#
# @author: laygin
#
# adapted from https://github.com/clovaai/deep-text-recognition-benchmark
#'''
import os, sys, re, six, math, lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
from utils.misc import ResizeNormalize, NormalizePAD
import config


logger = config.logger


def _accumulate(iterable, fn=lambda x, y: x + y):
    '''Return running totals'''
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class AlignCollate:
    def __init__(self, imgh=32, imgw=100, keep_ratio_with_pad=False):
        self.imgh = imgh
        self.imgw = imgw
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x:x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgw
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgh, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgh * ratio) > self.imgw:
                    resized_w = self.imgw
                else:
                    resized_w = math.ceil(self.imgh * ratio)

                resized_image = image.resize((resized_w, self.imgh), Image.BICUBIC)
                resized_images.append(transform(resized_image))
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform = ResizeNormalize((self.imgw, self.imgh))
            image_tensors = [transform(i) for i in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels


class LmdbDataset(Dataset):
    def __init__(self, root, imgh, imgw, batch_max_length, character, sensitive, rgb, data_filtering_off=False):
        self.root = root
        self.character = character
        self.sensitive = sensitive
        self.rgb = rgb
        self.imgH = imgh
        self.imgW = imgw

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            logger.error('can not create lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nsamples = int(txn.get('num-samples'.encode()))

            if data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nsamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nsamples):
                    index += 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > batch_max_length:
                        continue
                    out_of_char = f'[^{self.character}]'
                    label = label if self.sensitive else label.lower()
                    if re.search(out_of_char, label):
                        continue
                    self.filtered_index_list.append(index)
                self.nsamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                logger.error(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.rgb:
                    img = Image.new('RGB', (self.imgW, self.imgH))
                else:
                    img = Image.new('L', (self.imgW, self.imgH))
                label = '[dummy_label]'

            if not self.sensitive:
                label = label.lower()

            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return img, label


def hierarchical_dataset(root, imgh, imgw, batch_max_length,character,
                         sensitive=False, rgb=False, data_filtering_off=False, select_data='/'):
    dataset_list = []
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        if not dirnames:
            select_flag = False
            for select_d in select_data:
                if select_d in dirpath:
                    select_flag = True
                    break
            if select_flag:
                dataset = LmdbDataset(dirpath,
                                      imgh=imgh,
                                      imgw=imgw,
                                      batch_max_length=batch_max_length,
                                      character=character,sensitive=sensitive,rgb=rgb, data_filtering_off=data_filtering_off)
                logger.info(f'sub-directory:\t{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}')
                dataset_list.append(dataset)
    return ConcatDataset(dataset_list)


class BatchBalancedDataset:
    def __init__(self,
                 root,batch_max_length, character,
                 select_data,
                 batch_ratio,
                 batch_size,
                 total_data_usage_ratio,
                 workers,
                 sensitive=False, rgb=False, data_filtering_off=False,
                 imgh=32, imgw=100, keep_ratio_with_pad=False):
        assert len(select_data) == len(batch_ratio)

        _AlignCollate = AlignCollate(imgh=imgh, imgw=imgw, keep_ratio_with_pad=keep_ratio_with_pad)
        self.data_loader_list = []
        self.data_loader_iter_list = []
        batch_size_list = []
        total_batch_size = 0
        self.total_samples = 0
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            _batch_size = max(round(batch_size * float(batch_ratio_d)), 1)
            print('-' * 80)
            _dataset = hierarchical_dataset(root, imgh,imgw,batch_max_length,
                                            character,sensitive,rgb,data_filtering_off,
                                            select_data=[selected_d])
            total_number_dataset = len(_dataset)

            """
            total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            """
            number_dataset = int(total_number_dataset * float(total_data_usage_ratio))
            self.total_samples += number_dataset
            dataset_split = [number_dataset, total_number_dataset - number_dataset]

            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset-length: offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]

            logger.info(f'number total samples of {selected_d}: {total_number_dataset} x '
                        f'{total_data_usage_ratio} (usage_ratio)={len(_dataset)}')
            logger.info(f'number samples of {selected_d} per batch: {batch_size} x'
                        f'{batch_ratio_d}(batch ratio)={_batch_size}')

            batch_size_list.append(str(_batch_size))
            total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size, shuffle=True, num_workers=workers,
                collate_fn=_AlignCollate, pin_memory=True
            )
            self.data_loader_list.append(_data_loader)
            self.data_loader_iter_list.append(iter(_data_loader))

        print('-'*80)
        logger.info('Total_batch_size: ', '+'.join(batch_size_list), '=', str(total_batch_size))
        logger.info(f'Total samples: {self.total_samples}')
        self.batch_size = total_batch_size
        print('-' * 80)

    def __len__(self):
        return self.total_samples

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.data_loader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.data_loader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.data_loader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts
