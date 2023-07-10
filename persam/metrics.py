from common import *
from load import *

import cv2

def semseg_miou(gt_mask_dir:str,output_dir:str):
    running_sum = 0
    running_count = 0

    masks_in_dir = load_images_in_dir(gt_mask_dir)
    for gt_mask_name,gt_mask_path in masks_in_dir:
        output_mask_path = os.path.join(output_dir,gt_mask_name+".png")
        if os.path.exists(output_mask_path):
            output_mask = cv2.imread(output_mask_path) > 0
            gt_mask = cv2.imread(gt_mask_path) > 0

            intersection = np.logical_and(output_mask,gt_mask).sum()
            union = np.logical_or(output_mask,gt_mask).sum()

            iou = intersection / union if union > 0 else 1
            running_sum += iou
            running_count += 1
    
    miou = running_sum / running_count if running_count >0 else 1
    return miou

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Semantic segmentation metrics")
    parser.add_argument("--gt_dir_glob",type=str,help="Glob for ground truth mask directories",default="./data/Annotations/*")
    parser.add_argument("--output_dir_glob",type=str,help="Glob for predicted mask directories",default="./output/*")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()

    running_miou = 0
    num_dirs = 0

    gt_dirs = glob.glob(args.gt_dir_glob)

    for output_dir in glob.glob(args.output_dir_glob):
        gt_dir = gt_dirs.filter(lambda d: os.path.basename(d) == os.path.basename(output_dir))[0]

        miou = semseg_miou(gt_dir,output_dir)
        g = os.path.basename(gt_dir)
        print(f"{g}: {round(miou,3)}")
        num_dirs+=1
        running_miou+=miou
    
    avg_miou = running_miou / num_dirs
    print(f"Average mIoU: {round(avg_miou,3)}")
    