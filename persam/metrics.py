from common import *
from load import *

import cv2
import numpy as np

eps = 1e-10

def semseg_iou(gt_mask_dir: str, output_dir: str):
    running_intersection = 0
    running_union = 0
    running_iou = 0
    num_imgs = 0

    masks_in_dir = load_images_in_dir(gt_mask_dir)

    for gt_mask_name, gt_mask_path in masks_in_dir:
        output_mask_path = os.path.join(output_dir, gt_mask_name + ".png")
        if os.path.exists(output_mask_path):
            output_mask = cv2.imread(output_mask_path) > 0.5
            gt_mask = cv2.imread(gt_mask_path) > 0.5

            output_mask = np.any(output_mask,axis=2)
            gt_mask = np.any(gt_mask,axis=2)

            intersection = np.logical_and(output_mask, gt_mask).sum()
            union = np.logical_or(output_mask, gt_mask).sum()


            running_intersection += intersection
            running_union += union

            iou = intersection / (union + eps)
            running_iou += iou
            num_imgs += 1

    iou = running_intersection / (running_union + eps)

    return iou,running_iou/num_imgs


import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Semantic segmentation metrics")
    parser.add_argument(
        "--gt_dir_glob",
        type=str,
        help="Glob for ground truth mask directories",
        default="./data/Annotations/*",
    )
    parser.add_argument(
        "--output_dir_glob",
        type=str,
        help="Glob for predicted mask directories",
        default="./output/*",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    running_iou = 0
    running_iou_2 = 0
    num_dirs = 0

    gt_dirs = glob.glob(args.gt_dir_glob)

    for output_dir in glob.glob(args.output_dir_glob):
        matching_gt_dirs = [
            d for d in gt_dirs if os.path.basename(d) == os.path.basename(output_dir)
        ]
        assert len(matching_gt_dirs) == 1,f"Missing GT dir for {output_dir}. GT dirs are {gt_dirs}"
        gt_dir = matching_gt_dirs[0]

        iou,iou_2 = semseg_iou(gt_dir, output_dir)
        g = os.path.basename(gt_dir)
        print(f"{g}: {round(iou,3)}/{round(iou_2,3)}")
        num_dirs += 1
        running_iou += iou
        running_iou_2 += iou_2

    miou = running_iou / (num_dirs + eps)
    miou_2 = running_iou_2 / (num_dirs + eps)
    print(f"mIoU: {round(miou,3)}/{round(miou_2,3)}")
