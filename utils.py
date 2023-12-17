import math

import regex
import PIL
import numpy as np
import torch

from synthtiger import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):

        # character (str): set of the possible characters.
        list_special_token = [
            "[PAD]",
            "[UNK]",
            " ",
        ]  # [UNK] for unknown character, ' ' for space.
        # list_character = list(character)
        list_character = (
            opt.character  # we give list of character as a default. (especially for Hindi)
        )
        dict_character = list_special_token + list_character

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss, not same with space ' '.
            # print(i, char)
            self.dict[char] = i + 1

        self.character = [
            "[CTCblank]"
        ] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0).
        print(f"# of tokens and characters: {len(self.character)}")

    def encode(self, word_string, batch_max_length=25):
        """ convert word_list (string) into word_index.
        input:
            word_string: word labels of each image. [batch_size]
            batch_max_length: max length of word in the batch. Default: 25

        output:
            batch_word_index: word index list for CTCLoss. [batch_size, batch_max_length]
            batch_word_length: length of each word. [batch_size]
        """
        # word_length = [len(word) for word in word_string]
        batch_word_length = []

        # The index used for padding (=[PAD]) would not affect the CTC loss calculation.
        batch_word_index = torch.LongTensor(len(word_string), batch_max_length).fill_(
            self.dict["[PAD]"]
        )
        for i, word in enumerate(word_string):
            # For HIN/BEN/ARA, HIN/BEN/ARA chars are delimited with \t
            if regex.findall("[\p{Arabic}]", word):
                if "\t" in word:
                    word = word.split("\t")
                else:
                    # not to split 1 char into sub-chars. e.g. "list(मं)" splits मं into 2 sub-chars.
                    word = [word]
                # The order of Arabic labels in lmdb is right to left (human reading order),
                # so we reverse the label to align with the visual text order (left to right).
                word.reverse()

            elif regex.findall("[\p{InDevanagari}\p{InBengali}]", word):
                if "\t" in word:
                    word = word.split("\t")
                else:
                    # not to split 1 char into sub-chars. e.g. "list(मं)" splits मं into 2 sub-chars.
                    word = [word]
            else:
                word = list(word)

            # we calculate word_length after handling hindi chars.
            word_length = len(word)
            batch_word_length.append(word_length)
            word_idx = [
                self.dict[char] if char in self.dict else self.dict["[UNK]"]
                for char in word
            ]
            batch_word_index[i][:word_length] = torch.LongTensor(word_idx)

        return (
            batch_word_index.to(device),
            torch.IntTensor(batch_word_length).to(device),
        )

    def decode(self, word_index, word_length):
        """ convert word_index into word_string """
        word_string = []
        for idx, length in enumerate(word_length):
            word_idx = word_index[idx, :]

            char_list = []
            for i in range(length):
                # removing repeated characters and blank.
                if word_idx[i] != 0 and not (i > 0 and word_idx[i - 1] == word_idx[i]):
                    char_list.append(self.character[word_idx[i]])

            word = "".join(char_list)
            word_string.append(word)

        return word_string

    def decode_gt(self, word_index, word_length):
        """ convert word_index into word_string for gt"""
        word_string = []
        for idx, length in enumerate(word_length):
            word_idx = word_index[idx, :length]
            word = "".join([self.character[i] for i in word_idx])
            word = word.replace("[PAD]", "")
            word_string.append(word)

        return word_string


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):

        # character (str): set of the possible characters.
        # [SOS] (start-of-sentence token) and [EOS] (end-of-sentence token) for the attention decoder.
        list_special_token = [
            "[PAD]",
            "[UNK]",
            " ",
            "[SOS]",
            "[EOS]",
        ]  # [UNK] for unknown character, ' ' for space.
        # list_character = list(character)
        list_character = (
            opt.character  # we give list of character as a default. (especially for Hindi)
        )
        self.character = list_special_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

        print(f"# of tokens and characters: {len(self.character)}")

    def encode(self, word_string, batch_max_length=25):
        """ convert word_list (string) into word_index.
        input:
            word_string: word labels of each image. [batch_size]
            batch_max_length: max length of word in the batch. Default: 25

        output:
            batch_word_index : the input of attention decoder. [batch_size x (max_length+2)] +1 for [SOS] token and +1 for [EOS] token.
            batch_word_length : the length of output of attention decoder, which count [EOS] token also. [batch_size]
        """
        batch_max_length += 1

        # additional batch_max_length + 1 for [SOS] at first step.
        batch_word_index = torch.LongTensor(
            len(word_string), batch_max_length + 1
        ).fill_(self.dict["[PAD]"])
        batch_word_index[:, 0] = self.dict["[SOS]"]
        batch_word_length = []

        for i, word in enumerate(word_string):
            # For HIN/BEN/ARA, HIN/BEN/ARA chars are delimited with \t
            if regex.findall("[\p{Arabic}]", word):
                if "\t" in word:
                    word = word.split("\t")
                else:
                    # not to split 1 char into sub-chars. e.g. "list(मं)" splits मं into 2 sub-chars.
                    word = [word]
                # The order of Arabic labels in lmdb is right to left (human reading order),
                # so we reverse the label to align with the visual text order (left to right).
                word.reverse()

            elif regex.findall("[\p{InDevanagari}\p{InBengali}]", word):
                if "\t" in word:
                    word = word.split("\t")
                else:
                    # not to split 1 char into sub-chars. e.g. "list(मं)" splits मं into 2 sub-chars.
                    word = [word]
            else:
                word = list(word)

            # we calculate word_length after handling hindi chars.
            word_length = len(word) + 1  # +1 for [EOS] at end of sentence.
            batch_word_length.append(word_length)
            word.append("[EOS]")
            word_idx = [
                self.dict[char] if char in self.dict else self.dict["[UNK]"]
                for char in word
            ]
            batch_word_index[i][1 : 1 + word_length] = torch.LongTensor(
                word_idx
            )  # word_index[:, 0] = [SOS] token

        return (
            batch_word_index.to(device),
            torch.IntTensor(batch_word_length).to(device),
        )

    def decode(self, word_index, word_length):
        """ convert word_index into word_string """
        word_string = []
        for idx, length in enumerate(word_length):
            word_idx = word_index[idx, :length]
            word = "".join([self.character[i] for i in word_idx])
            word_string.append(word)

        return word_string


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


def adjust_learning_rate(optimizer, iteration, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr
    # stepwise lr schedule
    for milestone in opt.schedule:
        lr *= (
            opt.lr_drop_rate if iteration >= (float(milestone) * opt.num_iter) else 1.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = PIL.Image.fromarray(image_numpy)
    image_pil.save(image_path)
