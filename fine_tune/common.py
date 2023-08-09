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

from tqdm import tqdm

# notes for what these should look like / what they become:

# here's the predict() function from the SAM predictor:

from typing import Dict, Union, Tuple

# Coords = Tuple[ndarray,ndarray] # shape (N, 2), dtype float32; shape (N,), dtype bool
Box = ndarray # shape (4,), dtype float32, in XYXY format
BoolMask = ndarray # shape (H,W), dtype bool
FloatMask = ndarray # shape (H,W), dtype float32, in [0,1]

# Prompt = Dict[str,Union[Coords,Box,BoolMask]]

# using a dataclass instead:
from dataclasses import dataclass,asdict

@dataclass
class Prompt:
    points: ndarray = None
    labels: ndarray = None
    box: Box = None
    mask: BoolMask = None # TODO: also allow FloatMask

    gt_mask: BoolMask = None
    gt_masks: BoolMask = None

    multimask: bool = False

    def __post_init__(self):
        # assert self.points is not None or self.box is not None or self.mask is not None,"Prompt must have at least one of coords, box, or mask"

        if self.box is not None:
            assert self.box.shape == (4,),f"box must have shape (4,), not {self.box.shape}"

        # default to all points being foreground
        if self.points is not None and self.labels is None:
            self.labels = np.ones(len(self.points),dtype=bool)
        if self.points is not None or self.labels is not None:
            assert len(self.points.shape) == 2 and self.points.shape[1] == 2,f"points must have shape (N,2), not {self.points.shape}"
            assert len(self.labels.shape) == 1,f"points must have shape (N,), not {self.points.shape}"
            assert self.labels.dtype == bool,f"labels must have dtype bool, not {self.labels.dtype}"
            assert len(self.points) == len(self.labels),f"points and labels must have the same length, not {len(self.points)} and {len(self.labels)}"

        if self.mask is not None:
            assert len(self.mask.shape) == 2,f"mask must have shape (H,W), not {self.mask.shape}"
            assert self.mask.dtype == bool,f"mask must have dtype bool, not {self.mask.dtype}"

        assert (self.gt_mask is None) != (self.gt_masks is None),"Prompt must have exactly one of gt_mask or gt_masks"


import os

import hashlib
m = hashlib.sha256()

def hash_img_name(img_name: str) -> str:
    m.update(img_name.encode('utf-8'))
    return m.hexdigest()

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

should_cache = False

from torch.nn import functional as F

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

        images = self.dataset.images
        items = [(img_name,list(self.detections_to_prompts(images[img_name],dets))) for img_name,dets in tqdm(self.dataset.annotations.items())]
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
            embedding,input_size,original_size,resized_img = torch.load(embedding_path)
        else:
            predictor.set_image(img)
            embedding = predictor.features[0]

            assert torch.is_tensor(embedding),f"embedding is not a tensor, it is {type(embedding)}"
            assert len(embedding.shape) == 3,f"embedding shape is {embedding.shape}"

            original_size = predictor.original_size
            input_size = predictor.input_size

            resized_img = predictor.img.to(self.device)

            # save the embedding
            # torch.save((embedding,input_size,original_size,resized_img),embedding_path)
        
        sizing = (input_size,original_size)
        prompt_input,gt_masks = self.prompt_to_tensors(prompt,sizing)

        decoder_input = {
            "image_embeddings": embedding.to(self.device),
            **prompt_input,
        }

        return decoder_input, gt_masks, sizing, img, resized_img
    
    def prompt_to_tensors(self,prompt: Prompt, sizing: Tuple[torch.Tensor,torch.Tensor]):
        # mimic the predict() function from the SAM predictor
        input_size, original_size = sizing

        # Transform input prompts

        point_coords = prompt.points
        point_labels = prompt.labels

        mask_input = prompt.mask
        box = prompt.box

        coords_torch, labels_torch, box_torch, mask_input_torch, = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.predictor.transform.apply_coords(point_coords, original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.predictor.transform.apply_boxes(box, original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, None, :, :]
            mask_input_torch = F.interpolate(mask_input_torch, size=(256,256), mode="bilinear", align_corners=False)
        
        if point_coords is not None:
            points = (coords_torch, labels_torch)
        else:
            points = None
        
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.predictor.model.prompt_encoder(
            points=points,
            boxes=box_torch,
            masks=mask_input_torch,
        )

        decoder_input = {
            "image_pe": self.predictor.model.prompt_encoder.get_dense_pe().to(self.device),
            "sparse_prompt_embeddings": sparse_embeddings.to(self.device),
            "dense_prompt_embeddings": dense_embeddings.to(self.device),
            "multimask_output": prompt.multimask,
        }

        gt_mask = prompt.gt_mask
        gt_masks = prompt.gt_masks

        if gt_mask is not None:
            gt_masks = gt_mask[None,...]

        gt_masks = torch.as_tensor(gt_masks, dtype=torch.float, device=self.device)

        assert len(gt_masks.shape) == 3,f"gt_masks shape is {gt_masks.shape}"

        return decoder_input, gt_masks


    # could be i.e. filtering out only detections of a certain class
    # maybe is selecting a bunch of uniform points
    # maybe is training with bbox prompts
    # or is doing multi-point prompts
    # maybe uses one ground-truth mask, or doesn't have one specifically in mind
    # maybe uses a point and a segmentation mask
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> Union[List[Prompt],Iterable[Prompt]]:
        raise NotImplementedError


#
# Different training tasks
#

# treats the dataset as class-agnostic.
class SamBoxDataset(SamDataset):
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_cls,det_score,_ = det

            yield Prompt(
                box=det_box,
                gt_mask=det_mask,
                multimask=False,
            )

class SamPointDataset(SamDataset):
    def __init__(self, *args, points_per_mask=20, **kwargs):
        self.points_per_mask = points_per_mask
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_cls,det_score,_ = det

            mask_coords = np.nonzero(det_mask) # format: tuple of 1d arrays
            mask_coords = np.stack(mask_coords,axis=1) # format: 2d array
            mask_coords = mask_coords[np.random.permutation(mask_coords.shape[0])][:,::-1]

            point_coords = mask_coords[:self.points_per_mask]
            label = np.array([True],dtype=bool)
            for i in range(len(point_coords)):
                point_coord = point_coords[i]
                points = point_coord[None,...]

                yield Prompt(
                    points=points,
                    labels=label,
                    gt_mask=det_mask,
                    multimask=True
                )

from segment_anything.utils.amg import build_point_grid
class SamEverythingDataset(SamDataset):
    def __init__(self, points_per_side: int, top_k=3, *args, **kwargs):
        self.points_per_side = points_per_side
        self.top_k = top_k
        super().__init__(*args, **kwargs)
    
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:

        h,w,_ = img.shape
        input_size = np.array([w,h],dtype=float)

        # make a grid of points
        raw_sam_points = build_point_grid(self.points_per_side) # in [0,1]x[0,1] space

        # convert to pixel int coords
        raw_sam_points = raw_sam_points * input_size
        raw_sam_points = raw_sam_points.round().long()

        label = np.array([True],dtype=bool)
        for raw_sam_point in raw_sam_points:
            yield Prompt(
                points=raw_sam_point,
                labels=label,
                gt_masks=get_closest_dets(raw_sam_point, dets, self.top_k).mask,
                multimask=True,
            )

from math import ceil
class RandomPointDataset(SamDataset):
    def __init__(self, *args, points_per_img=50, top_k=3, **kwargs):
        self.points_per_img = points_per_img
        self.top_k = top_k
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> Iterable[Prompt]:

        h,w,_ = img.shape
        input_size = np.array([w,h],dtype=float)

        label = np.array([True],dtype=bool)
        for i in range(self.points_per_img):
            point = np.random.rand(1,2) * input_size

            yield Prompt(
                points=point,
                labels=label,
                gt_masks=get_closest_dets(point, dets, self.top_k).mask,
                multimask=True,
            )

from random import randint
class SamNextMaskDataset(SamDataset):
    def __init__(self, *args, secondary_prompter: SamDataset = None, splits_per_img=5, **kwargs):
        self.secondary_prompter = secondary_prompter
        self.splits_per_img = splits_per_img
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:
        for i in range(self.splits_per_img):
            # make random permutation of masks

            # mask_idxs = torch.randperm(len(dets))
            # or sort by area
            mask_idxs = np.argsort(dets.area)[::-1]

            new_dets = dets[mask_idxs]
            split_idx = randint(1,len(dets)-1)

            # make combined mask of all pre-split dets, this becomes the mask prompt
            # gt_dets are the ones after the split

            gt_dets = new_dets[split_idx:]
            combined_mask = get_combined_mask(img, new_dets[:split_idx])

            primary_prompt = Prompt(
                gt_masks=gt_dets.mask,
                mask=combined_mask
            )

            if self.secondary_prompter is None:
                return primary_prompt

            # enrich prompt with a secondary prompt
            secondary_prompt_gen = self.secondary_prompter.detections_to_prompts(img, gt_dets)

            for secondary_prompt in secondary_prompt_gen:
                # merge prompts
                if secondary_prompt.mask is None:
                    secondary_prompt.mask = primary_prompt.mask
                yield secondary_prompt

class SamSemSegDataset(SamDataset):
    def detections_to_prompts(self, img: ndarray, dets: Detections) -> List[Prompt] | Iterable[Prompt]:
        gt_mask = get_combined_mask(img, dets)
        yield Prompt(
            gt_mask=gt_mask,
            multimask=False,
        )

#
# UTILS
#

# def get_closest_dets(point: Tensor, dets: Detections, top_percent: float = 0.0) -> Detections:
#     # get closest detection to point, by avg L2 distance of bbox corners
#     xyxy = dets.xyxy
#     corners = torch.stack([
#         torch.stack([xyxy[:,0],xyxy[:,1]],dim=1),
#         torch.stack([xyxy[:,2],xyxy[:,3]],dim=1),
#         torch.stack([xyxy[:,0],xyxy[:,3]],dim=1),
#         torch.stack([xyxy[:,2],xyxy[:,1]],dim=1),
#     ],dim=1)

#     # avg L2 distance
#     l2_dist = torch.norm(corners - point[...,None],dim=1).mean(dim=0)

#     assert l2_dist.shape == (len(dets),), "l2 dist shape is wrong"
#     sorted_by_dist = torch.argsort(l2_dist)

#     k = ceil(len(dets) * top_percent / 100)
#     k = max(k,1)

#     return dets[sorted_by_dist[:k]]

# rewritten in numpy:

from supervision.dataset.utils import approximate_mask_with_polygons
def dets_to_polygonss(dets):
    polygons = []
    for _,det_mask,*_ in dets:
        polygons.append(approximate_mask_with_polygons(det_mask))
    return polygons

def get_distances(point, polygonss):
    point = point.squeeze()
    assert point.shape == (2,), "point shape is wrong"
    distances = np.zeros(len(polygonss), dtype=np.float32)
    for i,polygons in enumerate(polygonss):
        distances[i] = min([-cv2.pointPolygonTest(polygon, point, True) for polygon in polygons])
    
    distances = np.maximum(distances, 0)
    return distances

def get_closest_dets(point: np.ndarray, dets: Detections, top_k: int = 1) -> Detections:
    # get closest detection to point, by avg L2 distance of bbox corners
    # xyxy = dets.xyxy
    # corners = np.stack([
    #     np.stack([xyxy[:,0],xyxy[:,1]],axis=0),
    #     np.stack([xyxy[:,2],xyxy[:,3]],axis=0),
    #     np.stack([xyxy[:,0],xyxy[:,3]],axis=0),
    #     np.stack([xyxy[:,2],xyxy[:,1]],axis=0),
    # ],axis=0)

    # # avg L2 distance
    # l2_dist = np.linalg.norm(corners - point[...,None],axis=1).mean(axis=0)

    # assert l2_dist.shape == (len(dets),), "l2 dist shape is wrong"
    # sorted_by_dist = np.argsort(l2_dist)

    if top_k is None:
        return dets

    polygonss = dets_to_polygonss(dets)
    distances = get_distances(point, polygonss)
    sorted_by_dist = np.argsort(distances)

    return dets[sorted_by_dist[:top_k]]

def grow_mask(mask: np.ndarray, growth_radius: int) -> np.ndarray:
    # grow mask by growth_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (growth_radius, growth_radius))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return mask.astype(bool)

def grow_all_masks(dets:Detections,growth_radius: int):
    for i in range(len(dets)):
        mask = dets.mask[i]
        dets.mask[i] =  grow_mask(mask, growth_radius=growth_radius)
    
def grow_dataset_masks(dataset:DetectionDataset,growth_radius: int):
    for dets in dataset.annotations.values():
        grow_all_masks(dets,growth_radius=growth_radius)

def get_combined_mask(img: np.ndarray, detections: Detections) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=bool)

    for detection in detections:
        _, det_mask, *_ = detection
        mask[det_mask.astype(bool)] = True

    return mask

eps = 1e-6
def get_max_iou_masks(gt_masks: Tensor, pred_masks: Tensor) -> Tuple[Tensor,Tensor]:
    # get pred-gt mask pairing with highest IoU

    assert len(pred_masks) > 0,f"pred_masks is empty"

    # gt_masks should never be empty
    if len(gt_masks) == 0:
        gt_masks = torch.zeros_like(pred_masks[0])[None,...]
    
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
