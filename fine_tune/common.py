from fine_tune.prompts import Prompt
from supervision import DetectionDataset,Detections
from segment_anything import SamPredictor

from .datasets import merge_many_datasets

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
import cv2

from tqdm import tqdm

from typing import Any, Dict, Union, Tuple,Iterable, List

import os

from .prompts import Prompt,ContextPair,ClassInfo

from torch.nn import functional as F

class SamDataset(Dataset):

    def __init__(
            self,
            dataset: DetectionDataset,
            predictor: SamPredictor,
        ):

        self.dataset = dataset
        self.predictor = predictor

        images = self.dataset.images
        items = [(img_name,list(self.detections_to_prompts(images[img_name],dets))) for img_name,dets in tqdm(self.dataset.annotations.items())]
        self.prompts = [(img_name,prompt) for img_name,prompts in items for prompt in prompts]


    def __len__(self):
        return len(self.prompts)
    
    def img_to_embedding(self, img: np.ndarray):
        raise Exception("img_to_embedding is deprecated.")

        if self.cache_embeddings:
            img_name = int(img.sum())
            embedding_path = os.path.join(self.cache_dir,f"{img_name}.pt")

            if os.path.exists(embedding_path):
                info = torch.load(embedding_path, map_location="cpu")
                return info

        predictor = self.predictor

        predictor.set_image(img)
        embedding = predictor.features[0]

        assert torch.is_tensor(embedding),f"embedding is not a tensor, it is {type(embedding)}"
        assert len(embedding.shape) == 3,f"embedding shape is {embedding.shape}"

        original_size = predictor.original_size
        input_size = predictor.input_size

        unresized_img = torch.as_tensor(img)
        unresized_img = unresized_img.permute(2, 0, 1).contiguous()[None, :, :, :]
        unresized_img = (unresized_img - predictor.model.pixel_mean) / predictor.model.pixel_std

        resized_img = predictor.resized_img

        ret = embedding,(input_size,original_size),(unresized_img, resized_img)

        if self.cache_embeddings:
            torch.save(ret, embedding_path)

        return ret
    
    def ctx_to_embeddings(self, ctx: List[ContextPair]):
        raise Exception("ctx_to_embedding is deprecated.")

        if ctx is None:
            return {
                "context_embeddings": None,
            }
        assert len(context) == 1, "context size must be 1 for now"
        contexts = []
        for context in context:
            img = context.img
            mask = context.mask

            ctx_mask_input_torch = torch.as_tensor(mask, dtype=torch.float)
            ctx_mask_input_torch = ctx_mask_input_torch[None, None, :, :]
            ctx_mask_input_torch = F.interpolate(ctx_mask_input_torch, size=(256,256), mode="bilinear", align_corners=False)

            # encode with prompt encoder
            _, ctx_dense_embeddings = self.predictor.model.prompt_encoder(
                masks=mask[None,None,...],
            )

            ctx_embedding,*_ = self.img_to_embedding(img)
            ctx_embedding += self.predictor.model.prompt_encoder.context_embed

            ctx_embedding += ctx_dense_embeddings

            contexts.append(ctx_embedding)
        
        context_torch = torch.stack(contexts,dim=0)
        assert len(context_torch.shape) == 4,f"context_torch shape is {context_torch.shape}"

        return {
            "context_embeddings": context_torch,
        }

    def cls_to_tensors(self, gt_cls: np.ndarray, gt_clss: np.ndarray):
        if gt_cls is not None:
            gt_clss = gt_cls[None,...]

        if gt_clss is not None:
            gt_clss = torch.as_tensor(gt_clss, dtype=torch.float).to(torch.int64)
            # one-hot
            gt_clss_one_hot = F.one_hot(gt_clss, num_classes=len(self.dataset.classes)).to(torch.float32)

            return {
                "gt_cls": gt_clss,
                "gt_cls_one_hot": gt_clss_one_hot,
            }
        else:
            return None

    def __getitem__(self, idx:int):

        img_name,prompt = self.prompts[idx]
        # print(img_name)

        img = self.dataset.images[img_name]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs,sizes = self.predictor.preprocess_image(img)

        prompt_input,gt_masks = self.prompt_to_tensors(prompt,sizes)

        gt_cls_info = self.cls_to_tensors(prompt.gt_cls,prompt.gt_clss)

        gt_info = {
            "masks": gt_masks,
            "cls": gt_cls_info,
        }

        return prompt_input,gt_info, gt_cls_info, imgs,sizes, prompt
    
    @staticmethod
    def to_device(batch, device):
        prompt_input,gt_info, gt_cls_info, imgs,sizes, prompt = batch

        # move all to device
        prompt_input = {k:v.to(device) if torch.is_tensor(v) else v for k,v in prompt_input.items()}
        gt_info = {k:v.to(device) if torch.is_tensor(v) else v for k,v in gt_info.items()}
        gt_cls_info = {k:v.to(device) for k,v in gt_cls_info.items()} if gt_cls_info is not None else None
        imgs = tuple(img.to(device) if torch.is_tensor(img) else img for img in imgs)

        return prompt_input,gt_info, gt_cls_info, imgs,sizes, prompt

    def prompt_to_tensors(self,prompt: Prompt, sizing: Tuple[torch.Tensor,torch.Tensor]):

        # mimic the predict() function from the SAM predictor
        original_size,input_size = sizing

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
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.predictor.transform.apply_boxes(box, original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float)
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
            "image_pe": self.predictor.model.prompt_encoder.get_dense_pe(),
            "sparse_prompt_embeddings": sparse_embeddings,
            "dense_prompt_embeddings": dense_embeddings,
            "multimask_output": prompt.multimask,
        }

        gt_mask = prompt.gt_mask
        gt_masks = prompt.gt_masks

        if gt_mask is not None:
            gt_masks = gt_mask[None,...]

        gt_masks = torch.as_tensor(gt_masks, dtype=torch.float)

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
            det_box,det_mask,det_score,det_cls,_ = det

            assert det_box is not None, "det_box is None"

            yield Prompt(
                box=det_box,
                gt_mask=det_mask,
                multimask=False,
                gt_cls=det_cls,
            )

class SamPointDataset(SamDataset):
    def __init__(self, *args, points_per_mask=20, multimask=False, **kwargs):
        self.points_per_mask = points_per_mask
        self.multimask = multimask
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_score,det_cls,_ = det

            mask_coords = np.nonzero(det_mask) # format: tuple of 1d arrays
            mask_coords = np.stack(mask_coords,axis=1) # format: 2d array
            mask_coords = mask_coords[np.random.permutation(mask_coords.shape[0])][:,::-1]

            points_per_mask = self.points_per_mask if type(self.points_per_mask) == int else self.points_per_mask[det_cls.item()]

            point_coords = mask_coords[:points_per_mask]
            label = np.array([True],dtype=bool)
            for i in range(len(point_coords)):
                point_coord = point_coords[i]
                points = point_coord[None,...]

                yield Prompt(
                    points=points,
                    labels=label,
                    gt_mask=det_mask,
                    multimask=self.multimask,
                    gt_cls=det_cls,
                )

from segment_anything.utils.amg import build_point_grid
class SamEverythingDataset(SamDataset):
    def __init__(self, *args, points_per_side: int , top_k=3, **kwargs):
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
        raw_sam_points = raw_sam_points.round().astype(int)

        label = np.array([True],dtype=bool)
        for raw_sam_point in raw_sam_points:

            closest_dets = get_closest_dets(raw_sam_point, dets, self.top_k)

            yield Prompt(
                points=raw_sam_point[None,:],
                labels=label,
                gt_masks=closest_dets.mask,
                multimask=True,
                gt_clss=closest_dets.class_id,
                mask_loss=False,
            )

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

            closest_dets = get_closest_dets(point, dets, self.top_k)

            yield Prompt(
                points=point,
                labels=label,
                gt_masks=closest_dets.mask,
                multimask=True,
                gt_clss=closest_dets.class_id,
                mask_loss=False,
            )

from random import randint
class SamNextMaskDataset(SamDataset):
    def __init__(self, *args, secondary_prompter: SamDataset = None, splits_per_img=5, **kwargs):
        self.secondary_prompter = secondary_prompter
        self.splits_per_img = splits_per_img
        super().__init__(*args, **kwargs)
    def detections_to_prompts(self, img: np.ndarray, dets: Detections) -> List[Prompt]:
        if len(dets) == 0:
            return []
        for i in range(self.splits_per_img):
            # make random permutation of masks

            # mask_idxs = torch.randperm(len(dets))
            # or sort by area
            mask_idxs = np.argsort(dets.area)[::-1]

            new_dets = dets[mask_idxs]
            split_idx = randint(0,len(dets)-1)

            # make combined mask of all pre-split dets, this becomes the mask prompt
            # gt_dets are the ones after the split

            gt_dets = new_dets[split_idx:]
            combined_mask = get_combined_mask(img, new_dets[:split_idx])

            primary_prompt = Prompt(
                gt_masks=gt_dets.mask,
                mask=combined_mask
            )

            if self.secondary_prompter is None:
                yield primary_prompt
                continue

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

import random
class SamInContextDataset(SamDataset):
    def __init__(
            self,
            dataset: DetectionDataset,
            predictor: SamPredictor,

            context_size: int = 1,
            upsampling_factor: int = 3,
            main_dataset: DetectionDataset = None,
            dummy: bool = False
            ):
        super().__init__(dataset, predictor)

        assert context_size == 1, "context_size must be 1 for now"
        self.context_size = context_size

        assert main_dataset is not None, "main_dataset must be supplied"

        self.main_dataset = main_dataset

        self.upsampling_factor = upsampling_factor

        self.positive_examples = list(img for img,dets in self.main_dataset.annotations.items() if len(dets) > 0)

        self.dummy = dummy
    
    def detections_to_prompts(self, img: ndarray, dets: Detections) -> Union[List[Prompt], Iterable[Prompt]]:
        for prompt in self.main_dataset.detections_to_prompts(img, dets):
            for i in range(self.upsampling_factor):

                if self.dummy:
                    context = [ContextPair(
                        img=img,
                        mask=get_combined_mask(img, dets),
                    )]
                else:
                    context_name = random.sample(self.positive_examples, self.context_size)

                    context = []
                    for context_name in context:
                        context_img = self.main_dataset.images[context_name]
                        context_dets = self.main_dataset.annotations[context_name]

                        merged_mask = get_combined_mask(context_img, context_dets)

                        context.append(ContextPair(
                            img=context_img,
                            mask=merged_mask,
                        ))

                yield Prompt(
                    **prompt._asdict(),
                    context=context,
                )

class SamComboDataset(SamDataset):
    def __init__(
            self,
            sam_datasets: List[SamDataset],
            *args,
        ):

        # merge datasets
        self.dataset = merge_many_datasets([sam_dataset.dataset for sam_dataset in sam_datasets])

        super().__init__(self.dataset,*args)

        self.prompts = []
        for dataset in sam_datasets:
            self.prompts.extend(dataset.prompts)

    def detections_to_prompts(self, img: ndarray, dets: Detections) -> List[Prompt] | Iterable[Prompt]:
        return []


class SamDummyMaskDataset(SamDataset):
    def detections_to_prompts(self, img: ndarray, dets: Detections) -> Union[List[Prompt], Iterable[Prompt]]:
        for det in dets:
            _,det_mask,*_ = det
            yield Prompt(
                mask=det_mask,
                gt_mask=det_mask,
                multimask=False,
            )

#
# UTILS
#

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

def iou(a: Tensor, b: Tensor) -> Tensor:
    intersection = torch.sum(torch.minimum(a,b),dim=(-2,-1))
    union = torch.sum(torch.maximum(a,b),dim=(-2,-1))
    return intersection / (union + eps)

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