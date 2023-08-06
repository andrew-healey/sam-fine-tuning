# DetectionDataset / torch Dataset utils

# torch Dataset format:
# __getitem__ returns a tuple (cls, embedding, points_prompt, box_prompt, mask_prompt, gt_masks, gt_mask?)--already in tensor format, having been processed by .encoder and .prompt_encoder.

# A common format for turning a DetectionDataset into a torch Dataset:
# a function from Detections to an iterable (with length) of tuples (cls, points_prompt, box_prompt, mask_prompt, gt_masks, gt_mask?)
# if gt_mask is None, then the gt_mask should be the highest-IoU gt_mask in gt_masks

from supervision import DetectionDataset,Detections

import torch
from torch import Tensor
from torch.utils.data import Dataset

import numpy as np
from numpy import ndarray

import cv2

from segment_anything import SamPredictor

from typing import Iterable, List

# notes for what these should look like / what they become:

# here's the predict() function from the SAM predictor:
"""
Predict masks for the given input prompts, using the currently set image.

Arguments:
    point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    model. Each point is in (X,Y) in pixels.
    point_labels (np.ndarray or None): A length N array of labels for the
    point prompts. 1 indicates a foreground point and 0 indicates a
    background point.
    box (np.ndarray or None): A length 4 array given a box prompt to the
    model, in XYXY format.
    mask_input (np.ndarray): A low resolution mask input to the model, typically
    coming from a previous prediction iteration. Has form 1xHxW, where
    for SAM, H=W=256.
    multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    return_logits (bool): If true, returns un-thresholded masks logits
    instead of a binary mask.
    attn_sim (torch.Tensor): A mask of embeddings similarity scores, used in attention.
    target_embedding (torch.Tensor): A target embedding, used in attention.
    high_res (bool): If true, returns high resolution masks.


Returns:
    (np.ndarray): The output masks in CxHxW format, where C is the
    number of masks, and (H, W) is the original image size.
    (np.ndarray): An array of length C containing the model's
    predictions for the quality of each mask.
    (np.ndarray): An array of shape CxHxW, where C is the number
    of masks and H=W=256. These low resolution logits can be passed to
    a subsequent iteration as mask input.
"""

from typing import Dict, Union, Tuple

Coords = Tuple[ndarray,ndarray] # shape (N, 2), dtype float32; shape (N,), dtype bool
Box = ndarray # shape (4,), dtype float32, in XYXY format
BoolMask = ndarray # shape (H,W), dtype bool
FloatMask = ndarray # shape (H,W), dtype float32, in [0,1]

Prompt = Dict[str,Union[Coords,Box,BoolMask]]

import os

import hashlib
m = hashlib.sha256()

def hash_img_name(img_name: str) -> str:
    m.update(img_name.encode('utf-8'))
    return m.hexdigest()

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

should_cache = False

class SamDataset(Dataset):

    def __init__(
            self,
            dataset: DetectionDataset,
            predictor: SamPredictor,
            device: str = default_device,
        ):

        self.dataset = dataset
        self.predictor = predictor

        self.device = device

        items = [(img_name,list(self.detections_to_prompts(dets))) for img_name,dets in self.dataset.annotations.items()]
        self.prompts = [(img_name,prompt) for img_name,prompts in items for prompt in prompts]

        # make tmp directory for feature embeddings
        if not os.path.exists('tmp'):
            os.mkdir('tmp')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx:int):

        img_name,prompt = self.prompts[idx]

        img = self.dataset.images[img_name]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # check if we've already computed the embedding
        clean_img_name = hash_img_name(img_name) # avoid unsafe characters and collisions
        embedding_path = os.path.join('tmp',clean_img_name+'.np')

        predictor = self.predictor

        if should_cache and os.path.exists(embedding_path):
            embedding,input_size,original_size = torch.load(embedding_path)
        else:
            predictor.set_image(img)
            embedding = predictor.features[0]

            assert torch.is_tensor(embedding),f"embedding is not a tensor, it is {type(embedding)}"
            assert len(embedding.shape) == 3,f"embedding shape is {embedding.shape}"

            original_size = predictor.original_size
            input_size = predictor.input_size

            # save the embedding
            # torch.save((embedding,input_size,original_size),embedding_path)

        # mimic the predict() function from the SAM predictor

        # Transform input prompts

        points = prompt.get('points',None)
        point_labels = point_coords = None
        if points is not None:
            point_coords,point_labels = points
            point_labels = point_labels.astype(bool)

            # point_coords = torch.from_numpy(point_coords).to(self.device)
            # point_labels = torch.from_numpy(point_labels).to(self.device)

        mask_input = prompt.get('mask',None)
        box = prompt.get('box',None)

        coords_torch, labels_torch, box_torch, mask_input_torch, = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = predictor.transform.apply_coords(point_coords, original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = predictor.transform.apply_boxes(box, original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        
        if point_coords is not None:
            points = (coords_torch, labels_torch)
        else:
            points = None
        
        multimask = prompt.get('multimask',False)

        # Embed prompts
        sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
            points=points,
            boxes=box_torch,
            masks=mask_input,
        )

        decoder_input = {
            "image_embeddings": embedding.to(self.device),
            "image_pe": predictor.model.prompt_encoder.get_dense_pe().to(self.device),
            "sparse_prompt_embeddings": sparse_embeddings.to(self.device),
            "dense_prompt_embeddings": dense_embeddings.to(self.device),
            "multimask_output": multimask,
        }

        gt_mask = prompt.get('gt_mask',None)
        if gt_mask is not None:
            gt_mask = torch.as_tensor(gt_mask, dtype=torch.float, device=self.device)
            assert len(gt_mask.shape) == 2,f"gt_mask shape is {gt_mask.shape}"

        gt_masks = prompt.get('gt_masks',None)
        if gt_masks is not None:
            gt_masks = torch.as_tensor(gt_masks, dtype=torch.float, device=self.device)
            assert len(gt_masks.shape) == 3,f"gt_masks shape is {gt_masks.shape}"

        assert (gt_mask is None) != (gt_masks is None), "either gt_mask or gt_masks must be supplied, but not both"

        return decoder_input, (gt_mask, gt_masks), (input_size, original_size)

    # could be i.e. filtering out only detections of a certain class
    # maybe is selecting a bunch of uniform points
    # maybe is training with bbox prompts
    # or is doing multi-point prompts
    # maybe uses one ground-truth mask, or doesn't have one specifically in mind
    # maybe uses a point and a segmentation mask
    def detections_to_prompts(self, dets: Detections) -> Union[List[Prompt],Iterable[Prompt]]:
        raise NotImplementedError

eps = 1e-6
def get_max_iou_masks(gt_mask: Tensor, gt_masks: Tensor, pred_masks: Tensor) -> Tuple[Tensor,Tensor]:
    # if gt_mask is not None:
    #     return gt_mask
    # else:
    #     # get mask with highest IoU

    #     intersections = torch.sum(torch.minimum(gt_masks,pred_mask[None,...]),dim=(1,2))
    #     unions = torch.sum(torch.maximum(gt_masks,pred_mask[None,...]),dim=(1,2))

    #     ious = intersections / (unions + eps)

    #     best_idx = torch.argmax(ious)
    #     return gt_masks[best_idx]
    
    # rewrite where there are multiple pred masks:
    # get pred-gt mask with highest IoU

    if gt_masks is None:
        gt_masks = gt_mask[None,...]
    
    reshaped_preds = pred_masks[None,...]
    reshaped_gts = gt_masks[:,None,...]

    intersections = torch.sum(torch.minimum(reshaped_gts,reshaped_preds),dim=(2,3))
    unions = torch.sum(torch.maximum(reshaped_gts,reshaped_preds),dim=(2,3))

    ious = intersections / (unions + eps)

    max_iou_per_gt, best_pred_idx_per_gt = ious.max(dim=1)
    best_gt_idx = max_iou_per_gt.argmax(dim=0)
    best_pred_idx = best_pred_idx_per_gt[best_gt_idx]

    best_iou = ious[best_gt_idx,best_pred_idx]

    # make sure it's the actual minimum
    assert torch.max(ious) == best_iou, "min-finding is wrong"

    return gt_masks[best_gt_idx], pred_masks[best_pred_idx], best_iou, best_pred_idx

# treats the dataset as class-agnostic.
class SamBoxDataset(SamDataset):
    def detections_to_prompts(self, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_cls,det_score,_ = det

            yield {
                'box': det_box,
                'gt_mask': det_mask,
            }

class SamPointDataset(SamDataset):
    def __init__(self, *args, points_per_mask=20, **kwargs):
        self.points_per_mask = points_per_mask
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_cls,det_score,_ = det

            # # make index of mask coords
            # mask_coords = torch.nonzero(torch.from_numpy(det_mask))
            # # get random permutation of mask coords
            # mask_coords = mask_coords[torch.randperm(mask_coords.shape[0])]

            # # pick 1 point
            # point_coord = mask_coords[0]
            # points = (point_coord, torch.tensor([True],dtype=torch.bool))

            # rewritten using numpy:

            mask_coords = np.nonzero(det_mask) # format: tuple of 1d arrays
            mask_coords = np.stack(mask_coords,axis=1) # format: 2d array
            mask_coords = mask_coords[np.random.permutation(mask_coords.shape[0])][:,::-1]

            point_coords = mask_coords[:self.points_per_mask]
            for i in range(len(point_coords)):
                point_coord = point_coords[i]
                points = (point_coord[None,...], np.array([True],dtype=bool))

                yield {
                    'points': points,
                    'gt_mask': det_mask,
                    'multimask': True
                }

from segment_anything.utils.amg import build_point_grid
class SamEverythingDataset(SamDataset):
    def __init__(self, points_per_side: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_per_side = points_per_side
    
    def detections_to_prompts(self, dets: Detections) -> List[Prompt]:
        # make a grid of points

        raw_sam_points = build_point_grid(self.points_per_side) # in [0,1]x[0,1] space

        # convert to pixel int coords
        raw_sam_points = raw_sam_points * self.input_size
        raw_sam_points = raw_sam_points.round().long()

        for raw_sam_point in raw_sam_points:
            yield {
                'points': (raw_sam_point,np.array([True],dtype=bool)),
            }