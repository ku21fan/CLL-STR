import sys
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo(opt):
    """ model configuration """
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt)
    else:
        converter = AttnLabelConverter(opt)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print("loading pretrained model from %s" % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(opt, mode="test")
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo,
        pin_memory=True,
    )

    # predict
    model.eval()
    with torch.no_grad():
        log = open(f"./log_demo_result.txt", "w")
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            if "CTC" in opt.Prediction:
                preds = model(image)
            else:
                # For max length prediction
                text_for_pred = (
                    torch.LongTensor(batch_size)
                    .fill_(converter.dict["[SOS]"])
                    .to(device)
                )
                preds = model(image, text_for_pred, is_train=False)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

            dashed_line = "-" * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f"{dashed_line}\n{head}\n{dashed_line}")
            log.write(f"{dashed_line}\n{head}\n{dashed_line}\n")

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(
                image_path_list, preds_str, preds_max_prob
            ):
                if "Attn" in opt.Prediction:
                    pred_EOS = pred.find("[EOS]")
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0

                print(f"{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}")
                log.write(f"{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n")

        log.write("\n")
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        required=True,
        help="path to image_folder which contains text images",
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model to evaluation"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="CRNN|SVTR")
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )

    """ Charcomb """
    parser.add_argument(
        "--PAD",
        action="store_true",
        help="whether to keep ratio then pad for image resize",
    )
    parser.add_argument(
        "--Aug",
        type=str,
        default="None",
        help="whether to use augmentation |None|Crop|Rot|",
    )

    opt = parser.parse_args()

    with open(f"charset/MLT19_charset.txt", "r", encoding="utf-8") as file:
        opt.character = file.read().splitlines()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # for convenience
    if opt.model_name == "CRNN":  # CRNN
        opt.Transformation = "None"
        opt.FeatureExtraction = "VGG"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "CTC"

    elif opt.model_name == "SVTR":  # SVTR
        opt.Transformation = "None"
        opt.FeatureExtraction = "SVTR"
        opt.SequenceModeling = "None"
        opt.Prediction = "CTC"

    if opt.num_gpu > 1:
        print(
            "For lab setting, check your GPU number, you should be missed CUDA_VISIBLE_DEVICES=0 or typo"
        )
        sys.exit()

    demo(opt)
