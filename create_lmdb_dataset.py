import os
import random

import fire
import lmdb
import cv2
import numpy as np
from tqdm import tqdm

from synthtiger import utils


def check_image_valid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(
    inputPath, gtFile, outputPath, checkValid=True, num_sample="all", tqdm_position=0,
):
    """ a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=60 * 2 ** 30)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    if num_sample == "all":
        nSamples = len(datalist)
    else:
        nSamples = int(num_sample)

    for i, data_line in tqdm(
        enumerate(datalist), total=nSamples, position=tqdm_position
    ):
        data_split = data_line.strip("\n").split("\t")
        imagePath = data_split[0]
        label = data_split[1]

        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not check_image_valid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


def createDataset_HIN(
    inputPath, gtFile, outputPath, checkValid=True, num_sample="all", tqdm_position=0,
):
    """ a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=60 * 2 ** 30)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    if num_sample == "all":
        nSamples = len(datalist)
    else:
        nSamples = int(num_sample)

    for i, data_line in tqdm(
        enumerate(datalist), total=nSamples, position=tqdm_position
    ):
        data_split = data_line.strip("\n").split("\t")
        imagePath = data_split[0]
        label = data_split[1]

        # For HIN/BEN/ARA, HIN/BEN/ARA chars are delimited with \t
        label = "\t".join(utils.split_text(label, reorder=True))

        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not check_image_valid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


if __name__ == "__main__":
    fire.Fire(createDataset)
