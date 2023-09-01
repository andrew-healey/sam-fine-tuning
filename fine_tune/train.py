"""
Setup:
- will be passed a Roboflow dataset ID - for now, assume it's an rf download.
- should train a mask decoder which can serve as both normal SAM and custom SAM
- no cost-saving or time-saving measures for now - ie batching, early stopping
- strictly scoped to projects with >40 images and >200 annotations
- should include SAM-fallback logic -- i.e. stability score/iou prediction/unsure of cls -> kick back to SAM
- macro threshold for using model:
    - if avg # of clicks per instance is < than SAM (aka model is mask-good), OR
    - if all classes have >75% on-axis of the confusion matrix (aka model is cls-good)
- micro threshold for using a mask&cls prediction:
    - model is mask-good AND cls-good
    - highest cls iou prediction is higher than for SAM prediction(s?), AND
    - user has enabled "use cls predictions" in the UI
- micro threshold for using a mask-only prediction:
    - model is mask-good
    - highest cls iou prediction is higher than for SAM prediction(s?)
- micro threshold for using a cls prediction:
    - model is cls-good
    - user has enabled "use cls predictions" in the UI
"""

"""
Todos:
- Clicks-per-instance benchmarking
- Confusion matrix benchmarking
- Automatically pick best pred-IoU threshold for SAM vs. custom SAM
- Attn masks + duplicate points: LHS can't attend to RHS, vice versa
- Simple Flask server for requesting+downloading a trained model
- ONNX export as a function
- Train loop as a function - no custom configs yet, just set defaults
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from typing import Union,List

import supervision as sv

from segment_anything import Sam,SamPredictor

import xxhash
import os
import shutil

#
# Classes
#

from .ladder_cnn import CNN_SAM
from .samed import LoRA_Tiny_Image_Encoder,LoRA_Mask_Decoder

class WrappedImageEncoder(nn.Module):
    def __init__(self,
                 predictor: SamPredictor,

                 use_cnn: bool=False,
                 resize_before_cnn: bool=True,

                 use_lora: bool=False,
                 lora_r: int=4,

                 use_patch_embed: bool=False,

                 cache: bool=True,
    ):
        super().__init__()
        self.predictor = predictor
        self.image_encoder = predictor.model.image_encoder
        self.use_cnn = use_cnn
        self.use_lora = use_lora
        self.use_patch_embed = use_patch_embed

        if self.use_cnn:
            self.cnn = CNN_SAM(resize_before_cnn=resize_before_cnn)
        
        if self.use_lora:
            self.lora = LoRA_Tiny_Image_Encoder(self.image_encoder, r=lora_r)
        
        self.can_cache_embeddings = cache and not self.use_lora
        if self.can_cache_embeddings:
            self.h = xxhash.xxh64()
            self.cache_dir = "./encoder_cache"
            # delete and recreate cache dir
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.mkdir(self.cache_dir)

    def hash(self,array:Union[np.ndarray,torch.Tensor])->int:
        # content-based hashing of a numpy array
        assert self.can_cache_embeddings, "Cannot hash embeddings if they are not cached"
        if isinstance(array, np.ndarray):
            self.h.reset()
            self.h.update(array)
            return self.h.intdigest()
        elif isinstance(array, torch.Tensor):
            return self.hash(array.cpu().detach().numpy())
        else:
            raise TypeError(f"Cannot hash {type(array)}")
    
    def pre_cnn_forward(self,imgs)->torch.Tensor:
        if self.can_cache_embeddings:
            hash = self.hash(imgs[0])
            if os.path.exists(f"{self.cache_dir}/{hash}.pt"):
                return torch.load(f"{self.cache_dir}/{hash}.pt")
        
        input_image, input_image_torch, input_image_resized = imgs

        input_image_final = self.predictor.model.preprocess(input_image_resized)
        features = self.image_encoder(input_image_final)[0]

        if self.can_cache_embeddings:
            torch.save(features,f"{self.cache_dir}/{hash}.pt")
        
        return features
    
    def forward(self,imgs)->torch.Tensor:
        features = self.pre_cnn_forward(imgs)
        if self.use_cnn:
            _,input_image_torch,input_image_resized = imgs
            features += self.cnn(input_image_torch, input_image_resized)
        
        return features
    
    def get_trainable_parameters(self):
        combined_params = []
        if self.use_cnn:
            combined_params += list(self.cnn.parameters())
        if self.use_lora:
            combined_params += list(self.lora.get_parameters())
        if self.use_patch_embed:
            combined_params += list(self.image_encoder.patch_embed.parameters())
        
        return combined_params

class WrappedMaskDecoder(nn.Module):
    def __init__(
            self,
            predictor: SamPredictor,

            ft:bool=False,

            lora:bool=False,
            lora_r:int=4,

            use_cls:bool=True,
    ):
        super().__init__()
        self.predictor = predictor
        self.mask_decoder = predictor.model.mask_decoder

        self.ft = ft
        self.lora = lora
        self.use_cls = use_cls

        self.mask_decoder = predictor.mask_decoder

        if self.lora:
            self.lora_mask_decoder = LoRA_Mask_Decoder(self.mask_decoder, r=lora_r)
        
        if self.use_cls:
            self.mask_decoder.add_cls_token(predictor.mask_decoder.num_classes)
    
    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)
    
    # note: this is included in postprocess because it involves control flow (dynamic resizing, argmaxing+dynamic indexing).
    def postprocess(self, low_res_masks, iou_predictions, sizes):
        original_size,input_size = sizes

        upscaled_masks = self.predictor.model.postprocess_masks(low_res_masks,input_size,original_size).squeeze(0)
        binary_masks = F.normalize(F.threshold(upscaled_masks, 0.0, 0))

        max_idx = torch.argmax(iou_predictions)

        pred_mask = upscaled_masks[max_idx]
        binary_mask = binary_masks[max_idx]

        return \
            (upscaled_masks,binary_masks), \
            (pred_mask,binary_mask)

    
    def get_parameters(self):
        if self.ft:
            return self.mask_decoder.parameters()

        combined_params = []
        if self.lora:
            combined_params += list(self.lora_mask_decoder.get_parameters())
        if self.use_cls:
            for param_set in [self.mask_decoder.cls_mask_tokens,self.mask_decoder.cls_iou_token,self.mask_decoder.cls_hypernetworks_mlps,self.mask_decoder.cls_iou_prediction_head]:
                combined_params += list(param_set.parameters())
        
        return combined_params

#
# Metrics
#

def confusion_matrix(predictor: nn.Module, valid_dataset: Dataset) -> torch.Tensor:
    """
    Returns a confusion matrix of shape (num_classes, num_classes).
    The rows are the ground truth classes, the columns are the predicted classes.
    """

    running_matrix = torch.zeros((predictor.mask_decoder.num_classes, predictor.num_classes), dtype=torch.int32)

    for 