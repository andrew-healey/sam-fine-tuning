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

from typing import Callable, Iterable, Tuple, List, Optional

DetsToEntries = Callable[[DetectionDataset], Iterable[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]

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

Coords = (ndarray,ndarray) # shape (N, 2), dtype float32; shape (N,), dtype bool
Box = ndarray # shape (4,), dtype float32, in XYXY format
BoolMask = ndarray # shape (H,W), dtype bool
FloatMask = ndarray # shape (H,W), dtype float32, in [0,1]

from typing import Dict, Union
Prompt = Dict[str,Union[Coords,Box,BoolMask]]

import os

import hashlib
m = hashlib.sha256()

def hash_img_name(img_name: str) -> str:
    m.update(img_name.encode('utf-8'))
    return m.hexdigest()

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        if os.path.exists(embedding_path):
            embedding = torch.load(embedding_path)
        else:
            self.predictor.set_image(img)
            embedding = self.predictor.features

            assert torch.is_tensor(embedding),f"embedding is not a tensor, it is {type(embedding)}"
            assert len(embedding.shape) == 3,f"embedding shape is {embedding.shape}"

            # save the embedding
            torch.save(embedding,embedding_path)

        # mimic the predict() function from the SAM predictor

        # Transform input prompts

        points = prompt.get('points',None)
        point_labels = point_coords = None
        if points is not None:
            point_coords,point_labels = points
            point_labels = point_labels.astype(np.bool)

        mask_input = prompt.get('mask',None)
        box = prompt.get('box',None)

        coords_torch, labels_torch, box_torch, mask_input_torch, = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        
        if point_coords is not None:
            points = (point_coords, labels_torch)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box_torch,
            masks=mask_input,
        )

        decoder_input = {
            "image_embeddings": embedding,
            "image_pe": self.predictor.model.prompt_encoder.get_dense_pe(),
            "sparse_prompt_embeddings": sparse_embeddings,
            "dense_prompt_embeddings": dense_embeddings,
            "multimask_output": False, # TODO make this configurable
        }

        # convert to cuda
        decoder_input = {k:v.to(self.device) for k,v in decoder_input.items()}

        gt_mask = prompt.get('gt_mask',None)
        if gt_mask is not None:
            gt_mask = torch.as_tensor(gt_mask, dtype=torch.float, device=self.device)
            assert len(gt_mask.shape) == 2,f"gt_mask shape is {gt_mask.shape}"

        gt_masks = prompt.get('gt_masks',None)
        if gt_masks is not None:
            gt_masks = torch.as_tensor(gt_masks, dtype=torch.float, device=self.device)
            assert len(gt_masks.shape) == 3,f"gt_masks shape is {gt_masks.shape}"

        assert (gt_mask is None) != (gt_masks is None), "either gt_mask or gt_masks must be supplied, but not both"

        return decoder_input, (gt_mask, gt_masks)

    # could be i.e. filtering out only detections of a certain class
    # maybe is selecting a bunch of uniform points
    # maybe is training with bbox prompts
    # or is doing multi-point prompts
    # maybe uses one ground-truth mask, or doesn't have one specifically in mind
    # maybe uses a point and a segmentation mask
    def detections_to_prompts(self, dets: Detections) -> Union[List[Prompt],Iterable[Prompt]]:
        raise NotImplementedError

# treats the dataset as class-agnostic.
class SamBoxDataset(SamDataset):
    def detections_to_prompts(self, dets: Detections) -> List[Prompt]:
        for det in dets:
            det_box,det_mask,det_cls,det_score,_ = det

            yield {
                'box': det_box,
                'gt_mask': det_mask,
            }