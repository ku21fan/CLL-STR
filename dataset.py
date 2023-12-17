import os
import sys
import re
import six
import time
import math
import random

import lmdb
import regex
from tqdm import tqdm
from natsort import natsorted
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


def hierarchical_dataset(root, opt, select_data="/", mode="train", data_type="label"):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset: {select_data[0]}"
    print(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, mode=mode)

                sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                print(sub_dataset_log)
                dataset_log += f"{sub_dataset_log}\n"
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


def concat_dataset(root, opt, dataset_list, mode="train"):
    """ select_data='/' contains all sub-directory of root directory """
    concat_dataset_list = []
    for data in dataset_list:
        dataset_log = f"dataset_root:    {root}\tsub-directory:\t{data}"
        print(dataset_log)
        dataset_log += "\n"

        dataset = LmdbDataset(os.path.join(root, data), opt, mode=mode)
        tmp_log = f"\t num samples: {len(dataset)}"
        print(tmp_log)
        dataset_log += f"{tmp_log}\n"

        if mode == "train" and data == "LAT" and opt.data_usage_LAT != "all":
            total_number_dataset = len(dataset)
            number_dataset = int(opt.data_usage_LAT)

            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            dataset, _ = [
                Subset(dataset, indices[offset - length : offset])
                for offset, length in zip(_accumulate(dataset_split), dataset_split)
            ]
            tmp_log = f"{len(dataset)} {data} samples are used for the experiment"
            print(tmp_log)
            dataset_log += f"{tmp_log}\n"

        elif (
            mode == "train" and "SynthMLT" in data and opt.data_usage_SynthMLT != "all"
        ):
            total_number_dataset = len(dataset)
            number_dataset = int(opt.data_usage_SynthMLT)

            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            dataset, _ = [
                Subset(dataset, indices[offset - length : offset])
                for offset, length in zip(_accumulate(dataset_split), dataset_split)
            ]
            tmp_log = f"{len(dataset)} {data} samples are used for the experiment"
            print(tmp_log)
            dataset_log += f"{tmp_log}\n"

        concat_dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(concat_dataset_list)
    print(f"num samples of concatenated_dataset: {len(concatenated_dataset)}")

    if mode == "train" and opt.data_usage != "all":
        total_number_dataset = len(concatenated_dataset)
        number_dataset = int(opt.data_usage)

        dataset_split = [number_dataset, total_number_dataset - number_dataset]
        indices = range(total_number_dataset)
        concatenated_dataset, _ = [
            Subset(concatenated_dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(dataset_split), dataset_split)
        ]
        tmp_log = f"{len(concatenated_dataset)} samples are used for the experiment"
        print(tmp_log)
        dataset_log += f"{tmp_log}\n"

    # for faster training, we multiply small datasets itself.
    if mode == "train" and len(concatenated_dataset) < 100000:
        print(
            f"{len(concatenated_dataset)} is too small, it is multiplied to over 100K"
        )
        print(
            "CAUTION: If you use 'epoch' for training, you have to remove this multiplication part"
        )
        multiple_times = int(100000 / len(concatenated_dataset))
        dataset_self_multiple = [concatenated_dataset] * multiple_times
        concatenated_dataset = ConcatDataset(dataset_self_multiple)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    def __init__(self, root, opt, mode="train"):

        self.root = root
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # print(self.env.info())
        self.txn = self.env.begin(write=False)

        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        self.nSamples = int(self.txn.get("num-samples".encode()))

        self.filtered_index_list = []
        for index in tqdm(range(self.nSamples), total=self.nSamples):
            index += 1  # lmdb starts with 1
            label_key = "label-%09d".encode() % index
            label = self.txn.get(label_key).decode("utf-8")

            # length filtering
            length_of_label = len(label)

            # For HIN/BEN/ARA, HIN/BEN/ARA chars are delimited with \t
            if regex.findall("[\p{InDevanagari}\p{InBengali}\p{Arabic}]", label):
                if (
                    length_of_label > 2 * opt.batch_max_length - 1
                ):  # to count "\t" between hindi chars.
                    # print("length check", label)
                    continue

            elif length_of_label > opt.batch_max_length:
                # print("length check", label)
                continue

            self.filtered_index_list.append(index)

        self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        label_key = "label-%09d".encode() % index
        label = self.txn.get(label_key).decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = self.txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        try:
            img = PIL.Image.open(buf).convert("RGB")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
            label = "[dummy_label]"

        return (img, label)


class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = PIL.Image.open(self.image_path_list[index]).convert("RGB")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class AlignCollate(object):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode

        if opt.PAD:
            self.transform = NormalizePAD(opt)
        else:
            if opt.Aug == "None" or mode != "train":
                self.transform = ResizeNormalize((opt.imgW, opt.imgH))
            else:
                self.transform = Text_augment(opt)

    def __call__(self, batch):
        images, labels = zip(*batch)

        image_tensors = [self.transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


class NormalizePAD(object):
    def __init__(self, opt, interpolation=PIL.Image.BICUBIC):
        self.opt = opt
        self.interpolation = interpolation
        self.padded_size = (3, self.opt.imgH, self.opt.imgW)  # 3 for RGB input channel
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w, h = img.size
        ratio = w / float(h)
        if math.ceil(self.opt.imgH * ratio) > self.opt.imgW:
            resized_w = self.opt.imgW
        else:
            resized_w = math.ceil(self.opt.imgH * ratio)

        img = img.resize((resized_w, self.opt.imgH), self.interpolation)

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.padded_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.padded_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.padded_size[2] - w)
            )

        return Pad_img
