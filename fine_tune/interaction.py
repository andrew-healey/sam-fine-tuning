

"""

Defines utils for benchmarking interactive segmentation performance.

- get next user click + next prompt
- get refinement prompt (i.e. just feeding in the same mask + same prompt)
- count # of clicks per instance - this means 1 + the # of refinement prompts needed for 90 or 95% IoU (or some similar threshold)

"""

import torch
from .prompts import Prompt
from dataclasses import replace
from typing import Optional,List
import numpy as np

def get_next_interaction(pred_mask:torch.Tensor,binary_mask:torch.Tensor,gt_mask_idx: int,prompt:Prompt,threshold:Optional[float]=None)->Prompt:

    binary_mask = binary_mask.detach().cpu().numpy()

    gt_binary_mask = prompt.gt_masks[gt_mask_idx] if prompt.gt_masks is not None else prompt.gt_mask

    # bool-ify the masks
    assert binary_mask.dtype == bool,"binary_mask should be bool"
    # binary_mask = binary_mask > 0
    assert gt_binary_mask.dtype == bool,"gt_binary_mask should be bool"
    # gt_binary_mask = gt_binary_mask > 0

    false = gt_binary_mask ^ binary_mask

    false_indices = np.argwhere(false)
    if false_indices.shape[0] == 0: return None

    rand_idx = np.random.randint(false_indices.shape[0])
    
    pt = false_indices[rand_idx:rand_idx+1,::-1] # convert from (y,x) to (x,y)
    label = gt_binary_mask[None,pt[0,1],pt[0,0]]

    # add to prompt
    new_point = np.concatenate([prompt.points,pt],axis=0) if prompt.points is not None else pt
    new_label = np.concatenate([prompt.labels,label],axis=0).astype(bool) if prompt.labels is not None else label.astype(bool)

    new_gt_mask = gt_binary_mask

    new_cls = prompt.gt_clss[gt_mask_idx] if prompt.gt_clss is not None else prompt.gt_cls

    new_mask = pred_mask.detach().cpu().numpy()

    ret = replace(prompt,points=new_point,labels=new_label,gt_mask=new_gt_mask,gt_cls=new_cls,mask=new_mask)
    ret.gt_masks = ret.gt_clss = None

    return ret

def get_refinement_prompt(pred_mask:torch.Tensor,gt_mask_idx: int,prompt:Prompt)->Prompt:
    gt_binary_mask = prompt.gt_masks[gt_mask_idx]

    new_cls = prompt.gt_clss[gt_mask_idx] if prompt.gt_clss is not None else prompt.gt_cls

    ret = replace(prompt,gt_mask=gt_binary_mask,gt_cls=new_cls)
    ret.gt_masks = ret.gt_clss = None

    # numpy-ify
    ret.mask = pred_mask.detach().cpu().numpy()

    return ret

from .common import SamDataset,to
from .binary_mask import get_max_iou_masks

from typing import Tuple

from random import randrange

from tqdm import tqdm
@torch.no_grad()
def get_ious_and_clicks(
        sam,
        valid_dataset:SamDataset,
        nums_clicks:List[int],
        use_cls:bool,
        device:torch.device
    )->List[Tuple[float,int]]:
    max_num_clicks = max(nums_clicks)

    clicks = [] # list of (iou,num_clicks) tuples

    prompt_sequences:List[Tuple[np.ndarray,List[Prompt]]] = []


    for batch in tqdm(valid_dataset):

        prompt_input,gt_info,gt_cls_info, imgs,sizes, prompt = batch = to(batch,device)
        gt_masks = gt_info["masks"]
        device = gt_masks.device

        my_seq = (imgs[0],[prompt])
        if randrange(10)==0:
            prompt_sequences.append(my_seq)

        encoder_output = sam.encoder.get_decoder_input(imgs,prompt)

        for click_idx in range(max_num_clicks):

            # only simulate a re-computation if the mask is incomplete
            if prompt is not None:
                prompt_input,gt_masks = to(valid_dataset.prompt_to_tensors(prompt,sizes),device)
                gt_cls_info = to(valid_dataset.cls_to_tensors(prompt),device)

                low_res_masks,iou_predictions,cls_low_res_masks,cls_iou_predictions = sam.decoder(**prompt_input,**encoder_output)


                if use_cls:
                    assert len(cls_low_res_masks) == 1,"cls_low_res_masks should be a list of length 1"
                    assert len(cls_low_res_masks.shape)==4,"cls_low_res_masks should have shape (1,num_classes,H,W)"

                    (upscaled_masks,binary_masks), max_idx = sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)

                    # get the most correct cls prediction
                    gt_binary_mask, binary_mask, max_iou, best_cls, best_det = get_max_iou_masks(gt_masks,binary_masks,gt_cls_info["gt_cls"],torch.arange(sam.cfg.data.num_classes).to(device))
                    pred_mask = cls_low_res_masks[0,best_cls]
                else:
                    assert len(low_res_masks) == 1,"low_res_masks should be a list of length 1"
                    assert len(low_res_masks.shape)==4,"low_res_masks should have shape (1,num_masks,H,W)"
                    (upscaled_masks,binary_masks), max_idx = sam.decoder.postprocess(low_res_masks,iou_predictions,sizes)

                    # get the most correct cls prediction
                    gt_binary_mask, binary_mask, max_iou, best_pred, best_det = get_max_iou_masks(gt_masks,binary_masks,None,None)
                    pred_mask = low_res_masks[0,best_pred]

            if click_idx+1 in nums_clicks:
                iou = max_iou.cpu().item()
                clicks.append((iou,click_idx+1))
            
            if click_idx == max_num_clicks-1: break

            if prompt is not None:
                prompt = get_next_interaction(pred_mask,binary_mask,best_det,prompt)

                my_seq[1].append(prompt)
    
    assert len(clicks) > 0,"No instances in dataset"

    return clicks, prompt_sequences