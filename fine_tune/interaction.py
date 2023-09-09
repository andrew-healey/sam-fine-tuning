

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

from .common import SamDataset,get_max_iou_masks,to

# TODO integrate this with the confusion matrix finder + general evaluation loop - not important now, but good for efficiency later
# TODO more urgent: include a switch to test with cls or without cls--i.e. ask "has custom SAM improved performance?"
def get_clicks_per_instance(sam,valid_dataset:SamDataset,threshold:float)->int:
    """
    Note: the batch has only one instance.
    Measure the number of clicks needed to get to a certain IoU threshold.
    """

    running_clicks = 0
    running_count = 0

    for batch in valid_dataset:

        prompt_input,gt_info,gt_cls_info, imgs,sizes, prompt = batch = to(batch,sam.device)

        assert sam.cfg.model.decoder.use_cls,"This function only works for cls models so far. TODO - add support for normal classless SAM."

        encoder_output = sam.encoder.get_decoder_input(imgs,prompt)

        while True:
            prompt_input = valid_dataset.prompt_to_tensors(prompt,sizes)
            low_res_masks,iou_predictions,cls_low_res_masks,cls_iou_predictions = sam.decoder(**prompt_input,**encoder_output)

            (upscaled_masks,binary_masks), max_idx = sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)
            running_clicks += 1

            # get the most correct cls prediction
            gt_binary_mask, binary_mask, max_iou, best_cls, best_det = get_max_iou_masks(gt_info["gt_masks"],binary_masks,gt_cls_info["gt_cls"],torch.arange(sam.cfg.model.decoder.num_classes).to(sam.device))

            prompt = get_next_interaction(binary_mask,best_det,prompt,threshold)
            if prompt is None: break
        
        running_count += 1
    
    assert running_count > 0,"No instances in dataset"
    
    return running_clicks / running_count

from tqdm import tqdm
@torch.no_grad()
def get_ious_at_click_benchmarks(
        sam,
        valid_dataset:SamDataset,
        nums_clicks:List[int],
        use_cls:bool,
        device:torch.device
    )->List[float]:
    max_num_clicks = max(nums_clicks)

    running_ious = [0] * len(nums_clicks)
    running_count = 0

    for batch in tqdm(valid_dataset):
        prompt_input,gt_info,gt_cls_info, imgs,sizes, prompt = batch = to(batch,device)
        gt_masks = gt_info["masks"]
        device = gt_masks.device

        encoder_output = sam.encoder.get_decoder_input(imgs,prompt)

        for click_idx in range(max_num_clicks):
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

            if click_idx in nums_clicks:
                iou_idx = nums_clicks.index(click_idx)
                running_ious[iou_idx] += max_iou.cpu().item()
            prompt = get_next_interaction(binary_mask,best_det,prompt)
        running_count += 1
    
    assert running_count > 0,"No instances in dataset"

    return [iou / running_count for iou in running_ious]