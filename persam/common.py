import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "segment-anything"))

from typing import Tuple, Optional, Dict, Any

from segment_anything import SamPredictor

import torch
from torch.nn import functional as F

import numpy as np

eps = 1e-10


# Assume the image is already loaded in the predictor
def get_mask_embed(
    predictor: SamPredictor, ref_mask: torch.Tensor, should_normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    feat_dims = ref_feat.shape[0:2]
    ref_mask = F.interpolate(ref_mask, size=feat_dims, mode="bilinear")
    ref_mask = ref_mask[0, 0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]

    if target_feat.shape[0] == 0:
        target_feat = target_embedding = torch.zeros(1, 256).cuda()
        return target_feat, target_embedding, feat_dims

    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0]

    target_embedding = target_feat_mean.unsqueeze(0)

    # Two modes: normalize or not
    # We use should_normalize for PerSAM and not for PerSAM_f
    if not should_normalize:
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
    else:
        target_feat = target_feat_mean.unsqueeze(0)

    target_feat = target_feat / (eps + target_feat.norm(dim=-1, keepdim=True))

    target_embedding = target_embedding.unsqueeze(0)

    return target_feat, target_embedding, feat_dims


def get_sim_map(predictor: SamPredictor, target_feat: torch.Tensor) -> torch.Tensor:
    test_feat = predictor.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / (eps + test_feat.norm(dim=0, keepdim=True))
    test_feat = test_feat.reshape(C, h * w)
    sim = target_feat @ test_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim, input_size=predictor.input_size, original_size=predictor.original_size
    ).squeeze()

    sim = (sim - sim.mean()) / (eps + torch.std(sim))
    sim = sim.sigmoid_()

    return sim


def sim_map_to_attn(sim_map: torch.Tensor) -> torch.Tensor:
    # Obtain the target guidance for cross-attention layers
    sim_map = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear"
    )
    attn_sim = sim_map.unsqueeze(0).flatten(3)

    return attn_sim


def get_extrema(
    sim_map: torch.Tensor, topk=1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Top-1 point selection
    w, h = sim_map.shape
    topk_xy = sim_map.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = sim_map.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label


def sim_map_to_points(
    sim_map: torch.Tensor, include_neg: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Positive-negative location prior
    topk_xy, topk_label, last_xy, last_label = get_extrema(sim_map, topk=1)

    if include_neg:
        topk_xy = np.concatenate([topk_xy, last_xy], axis=0)
        topk_label = np.concatenate([topk_label, last_label], axis=0)

    # Positive location prior
    return topk_xy, topk_label


def points_to_kwargs(points: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    topk_xy, topk_label = points
    return {"point_coords": topk_xy, "point_labels": topk_label}


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x / (eps + x.norm(dim=-1, keepdim=True))
    y = y / (eps + y.norm(dim=-1, keepdim=True))
    return x @ y.transpose(-1, -2)


def predict_mask_refined(
    predictor: SamPredictor,
    target_guidance: Dict[str, torch.Tensor],
    mask_picking_method: str,
    mask_picking_data: Optional[np.ndarray] = None,
    use_box: bool = True,
    **kwargs
) -> torch.Tensor:
    kwargs = {
        **kwargs,
        "high_res": True,
    }

    mask_dict = {}

    get_single_mask = mask_picking_method == "single"

    # First-step prediction
    masks, scores, logits, logits_high = predictor.predict(
        **kwargs, **target_guidance, multimask_output=not get_single_mask
    )

    for i in range(len(masks)):
        mask_dict[f"first_{i}"] = masks[i]

    def get_best_log_distance(arr: list[any], target: any):
        log_arr = torch.log(torch.tensor(arr))
        log_target = torch.log(torch.tensor(target))
        log_dist = torch.abs(log_arr.cpu() - log_target.cpu())
        return torch.argmin(log_dist)

    # Experiments!

    # -1 means best_idx is unset
    best_idx = -1

    if get_single_mask:
        best_idx = 0
    elif mask_picking_method in ["linear_combo", "best_idx", "best_idx_iou"]:
        logit_weights = mask_picking_data
        # Weighted sum of three-scale masks
        logits_high = logits_high * logit_weights[..., None, None]
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).cpu().detach().numpy()

        logit_weights_np = logit_weights.detach().cpu().numpy()

        logits = logits * logit_weights_np[..., None, None]
        logit = logits.sum(0)
    elif mask_picking_method == "area":
        areas = torch.tensor([masks[idx].sum().tolist() for idx in range(3)])
        ref_area = mask_picking_data
        best_idx = get_best_log_distance(areas, ref_area)
    elif mask_picking_method in ["bbox_area", "perimeter"]:
        boxes = [mask_to_box(masks[idx])[0] for idx in range(3)]
        widths = [box[2] - box[0] for box in boxes]
        heights = [box[3] - box[1] for box in boxes]

        if mask_picking_method == "bbox_area":
            ref_area = mask_picking_data

            areas = [width * height for width, height in zip(widths, heights)]
            best_idx = get_best_log_distance(areas, ref_area)
        elif mask_picking_method == "perimeter":
            ref_perimeter = mask_picking_data

            perimeters = [
                2 * (width + height) for width, height in zip(widths, heights)
            ]
            best_idx = get_best_log_distance(perimeters, ref_perimeter)
    elif mask_picking_method == "sam_embedding":
        sam_embedding, should_normalize = mask_picking_data
        torch_masks = torch.from_numpy(masks).to(torch.float).cuda()
        mask_embeds = [
            get_mask_embed(predictor, torch_masks[None, None, idx], should_normalize)[1]
            for idx in range(3)
        ]
        cosine_similarities = torch.stack(
            [cosine_similarity(sam_embedding, mask_embed) for mask_embed in mask_embeds]
        )
        best_idx = torch.argmax(cosine_similarities)
    elif mask_picking_method == "clip_embedding":
        raise NotImplementedError()
    elif mask_picking_method == "sim":
        # get average similarity score across each mask
        sim_map = mask_picking_data
        torch_masks = torch.from_numpy(masks).to(torch.float).cuda()
        filtered_maps = [sim_map * mask for mask in torch_masks]
        sim_scores = torch.stack(
            [torch.mean(filtered_map) for filtered_map in filtered_maps]
        )
        best_idx = torch.argmax(sim_scores)
    elif mask_picking_method == "max_score":
        best_idx = torch.argmax(torch.tensor(scores))

    # Only extract "best" mask if best_idx is set
    if best_idx >= 0:
        mask = masks[best_idx]
        logit = logits[best_idx]

    # Cascaded Post-refinement-1
    box = mask_to_box(mask) if use_box else None

    masks, scores, logits, _ = predictor.predict(
        **kwargs, box=box, mask_input=logit[None, :, :], multimask_output=True
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    logit = logits[best_idx]

    for i in range(len(masks)):
        mask_dict[f"post1_{i}"] = masks[i]

    # Cascaded Post-refinement-2
    box = mask_to_box(mask)
    masks, scores, logits, _ = predictor.predict(
        **kwargs, box=box, mask_input=logit[None, :, :], multimask_output=True
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    for i in range(len(masks)):
        mask_dict[f"post2_{i}"] = masks[i]

    return mask,mask_dict


def mask_to_box(mask: torch.Tensor) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0 or len(y) == 0:
        return np.array([0, 0, 0, 0])
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    input_box = np.array([x_min, y_min, x_max, y_max])
    return input_box[None, :]
