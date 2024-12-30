import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from wandb.wandb_torch import torch

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def compute_f1_score(gt_dir, pred_dir, image_ids, num_classes):

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for image_id in image_ids:
        gt_image = np.array(Image.open(os.path.join(gt_dir, image_id + ".png")))
        pred_image = np.array(Image.open(os.path.join(pred_dir, image_id + ".png")))

        for cls in range(num_classes):
            tp[cls] += np.sum((pred_image == cls) & (gt_image == cls))
            fp[cls] += np.sum((pred_image == cls) & (gt_image != cls))
            fn[cls] += np.sum((pred_image != cls) & (gt_image == cls))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-7)

    return f1_scores


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 2
    name_classes = ["background", "road"]
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")

        # 计算F1分数
        print("Get F1 score.")
        f1_scores = compute_f1_score(gt_dir, pred_dir, image_ids, num_classes)
        for i, name in enumerate(name_classes):
            print(f"Class {name}: F1 Score = {f1_scores[i]:.4f}")
        print("Get F1 score done.")

        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
