from typing import Tuple
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.nn.functional import interpolate

from .common import iou

def get_max_iou_masks(gt_masks: Tensor, pred_masks: Tensor, gt_cls: Tensor=None, pred_cls: Tensor=None) -> Tuple[Tensor,Tensor, Tensor, int]:
    # get pred-gt mask pairing with highest IoU

    assert len(pred_masks) > 0,f"pred_masks is empty"

    # gt_masks should never be empty
    if len(gt_masks) == 0:
        gt_masks = torch.zeros_like(pred_masks[0])[None,...]
    
    reshaped_preds = pred_masks[None,...]
    reshaped_gts = gt_masks[:,None,...]

    ious = iou(reshaped_preds, reshaped_gts)

    if gt_cls is not None and pred_cls is not None:
        reshaped_pred_cls = pred_cls[None,...]
        reshaped_gt_cls = gt_cls[:,None,...]

        ious[reshaped_pred_cls != reshaped_gt_cls] = -1

    max_iou_per_gt, best_pred_idx_per_gt = ious.max(dim=1)
    best_gt_idx = max_iou_per_gt.argmax(dim=0)
    best_pred_idx = best_pred_idx_per_gt[best_gt_idx]

    assert len(ious.shape) == 2,f"ious shape is {ious.shape}"
    best_iou = ious[best_gt_idx,best_pred_idx]

    # make sure it's the actual minimum
    assert torch.max(ious) == best_iou, "min-finding is wrong"

    return gt_masks[best_gt_idx], pred_masks[best_pred_idx], best_iou, best_pred_idx, best_gt_idx

def get_mask_stability(low_res_mask,offset,threshold=0)->float:
    # iou of (low_res_mask + offset) > 0 and (low_res_mask - offset) > 0

    up_mask = (low_res_mask + offset) > threshold
    down_mask = (low_res_mask - offset) > threshold

    return iou(up_mask,down_mask)

import numpy as np

# dynamically pick threshold to improve mask stability
# hopefully this might get rid of gridiron artifacts
def get_binary_mask_dynamic(low_res_mask,offset,threshold=0)->float:
    # masks are [low_res_mask + offset * i > 0 for i in arange(-2,2,offset)]
    offsets = np.arange(-2,2,offset)
    masks = [(low_res_mask + offset) > threshold for offset in offsets]

    naive_mask = low_res_mask > threshold
    valid_masks = [iou(naive_mask,mask) > 0.5 for mask in masks]

    # get siblingwise ious
    ious = [iou(masks[i],masks[i+1]) for i in range(len(masks)-1)]
    filtered_ious = torch.stack([iou if valid_masks[i] else iou*0 for i,iou in enumerate(ious)])

    argmax = torch.argmax(filtered_ious)
    # best offset = avg of the two masks
    offset = (offsets[argmax] + offsets[argmax+1]) / 2
    return (low_res_mask + offset) > threshold

def binarize_dynamic(upscaled_masks:Tensor)->Tensor:
    out = torch.zeros_like(upscaled_masks).bool()
    for i,mask in enumerate(upscaled_masks):
        out[i] = get_binary_mask_dynamic(mask,0.1)
    return out