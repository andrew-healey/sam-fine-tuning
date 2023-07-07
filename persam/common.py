
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),"..","segment-anything"))

from typing import Tuple,Optional,Dict

from segment_anything import SamPredictor

import torch
from torch.nn import functional as F

import numpy as np

# Assume the image is already loaded in the predictor
def get_mask_embed(predictor:SamPredictor,ref_mask:torch.Tensor,should_normalize:bool=True)->Tuple[torch.Tensor,torch.Tensor]:
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]

    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0]

    target_embedding = target_feat_mean.unsqueeze(0)

    # Two modes: normalize or not
    # We use should_normalize for PerSAM and not for PerSAM_f
    if should_normalize:
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    else:
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
    
    target_embedding = target_embedding.unsqueeze(0)

    return target_feat,target_embedding

def get_sim_map(predictor:SamPredictor,target_feat:torch.Tensor)->torch.Tensor:
    test_feat = predictor.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)
    sim = target_feat @ test_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim,
        input_size=predictor.input_size,
        original_size=predictor.original_size
    ).squeeze()

    return sim

def sim_map_to_attn(sim_map:torch.Tensor)->torch.Tensor:
    # Obtain the target guidance for cross-attention layers
    sim_map = (sim_map - sim_map.mean()) / torch.std(sim_map)
    sim_map = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim_map.sigmoid_().unsqueeze(0).flatten(3)

    return attn_sim

def get_extrema(sim_map:torch.Tensor, topk=1)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # Top-1 point selection
    w, h = sim_map.shape
    topk_xy = sim_map.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = sim_map.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

def sim_map_to_points(sim_map:torch.Tensor)->Tuple[np.ndarray,np.ndarray]:
    # Positive-negative location prior
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = get_extrema(sim_map, topk=1)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    # Positive location prior
    return topk_xy, topk_label

def points_to_kwargs(points:Tuple[np.ndarray,np.ndarray])->Dict[str,np.ndarray]:
    topk_xy, topk_label = points
    return {
        "point_coords": topk_xy,
        "point_labels": topk_label
    }

def predict_mask_refined(predictor:SamPredictor,target_guidance:Dict[str,torch.Tensor]={},logit_weights:Optional[np.ndarray]=None,use_box:bool=True,**kwargs)->torch.Tensor:
    # First-step prediction
    masks, scores, logits, logits_high = predictor.predict(
                **kwargs,
                **target_guidance,
                multimask_output=True)

    if logit_weights is None:
        best_idx = 0
        mask = masks[best_idx]
        logit = logits[best_idx]
    else:
        # Weighted sum three-scale masks
        logits_high = logits_high * logit_weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logit_weights_np = logit_weights.detach().cpu().numpy()

        logits = logits * logit_weights_np[..., None]
        logit = logits.sum(0)

    # Cascaded Post-refinement-1
    box = mask_to_box(mask) if use_box else None

    masks, scores, logits, _ = predictor.predict(
        **kwargs,
        box=box,
        mask_input=logit[None, :, :],
        multimask_output=True)
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    logit = logits[best_idx]

    # Cascaded Post-refinement-2
    box = mask_to_box(mask)
    masks, scores, logits, _ = predictor.predict(
        **kwargs,
        box=box,
        mask_input=logit[None,:,:],
        multimask_output=True)
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    return mask

def mask_to_box(mask:torch.Tensor)->np.ndarray:
    y, x = np.nonzero(mask)
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    return input_box[None,:]