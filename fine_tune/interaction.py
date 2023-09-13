

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

def get_next_interaction(binary_mask:torch.Tensor,gt_mask_idx: int,prompt:Prompt,threshold:Optional[float]=None)->Prompt:

    binary_mask = binary_mask.detach().cpu().numpy()

    gt_binary_mask = prompt.gt_masks[gt_mask_idx] if prompt.gt_masks is not None else prompt.gt_mask

    # bool-ify the masks
    binary_mask = binary_mask > 0
    gt_binary_mask = gt_binary_mask > 0

    false_negatives = gt_binary_mask & ~binary_mask
    false_positives = ~gt_binary_mask & binary_mask

    iou = (binary_mask & gt_binary_mask).sum() / (binary_mask | gt_binary_mask).sum()
    if threshold is not None and iou >= threshold: return None

    # pick random next click
    fn_indices  = np.argwhere(false_negatives)
    fp_indices  = np.argwhere(false_positives)

    # label fn_indices with 1, fp_indices with 0
    fn_indices = np.concatenate([fn_indices,np.ones((fn_indices.shape[0],1),dtype=int)],axis=1)
    fp_indices = np.concatenate([fp_indices,np.zeros((fp_indices.shape[0],1),dtype=int)],axis=1)

    # concatenate indices
    indices = np.concatenate([fn_indices,fp_indices],axis=0)

    if indices.shape[0] == 0: return None

    rand_idx = np.random.randint(indices.shape[0])
    
    pt = indices[rand_idx,:2][::-1][None,...] # convert from (y,x) to (x,y)
    label = indices[rand_idx,2][None,...]

    # add to prompt
    new_point = np.concatenate([prompt.points,pt],axis=0) if prompt.points is not None else pt
    new_label = np.concatenate([prompt.labels,label],axis=0).astype(bool) if prompt.labels is not None else label.astype(bool)

    new_mask = gt_binary_mask

    new_cls = prompt.gt_clss[gt_mask_idx] if prompt.gt_clss is not None else prompt.gt_cls

    ret = replace(prompt,points=new_point,labels=new_label,gt_mask=new_mask,gt_cls=new_cls)
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

    for batch in tqdm(valid_dataset):
        prompt_input,gt_info,gt_cls_info, imgs,sizes, prompt = batch = to(batch,device)
        gt_masks = gt_info["masks"]
        device = gt_masks.device

        encoder_output = sam.encoder.get_decoder_input(imgs,prompt)

        for click_idx in range(max_num_clicks):

            # only simulate a re-computation if the mask is incomplete
            if prompt is not None:
                prompt_input,gt_masks = to(valid_dataset.prompt_to_tensors(prompt,sizes),device)
                gt_cls_info = to(valid_dataset.cls_to_tensors(prompt),device)

                low_res_masks,iou_predictions,cls_low_res_masks,cls_iou_predictions = sam.decoder(**prompt_input,**encoder_output)

                if use_cls:
                    (upscaled_masks,binary_masks), max_idx = sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)

                    # get the most correct cls prediction
                    gt_binary_mask, binary_mask, max_iou, best_cls, best_det = get_max_iou_masks(gt_masks,binary_masks,gt_cls_info["gt_cls"],torch.arange(sam.cfg.data.num_classes).to(device))
                else:
                    (upscaled_masks,binary_masks), max_idx = sam.decoder.postprocess(low_res_masks,iou_predictions,sizes)

                    # get the most correct cls prediction
                    gt_binary_mask, binary_mask, max_iou, best_pred, best_det = get_max_iou_masks(gt_masks,binary_masks,None,None)

            if click_idx+1 in nums_clicks:
                iou = max_iou.cpu().item()
                clicks.append((iou,click_idx+1))
            prompt = get_next_interaction(binary_mask,best_det,prompt)
    
    assert len(clicks) > 0,"No instances in dataset"

    return clicks