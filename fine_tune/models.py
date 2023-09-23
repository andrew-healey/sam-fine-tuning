import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from typing import Union,List

import supervision as sv

from segment_anything import SamPredictor

import xxhash
import os
import shutil

from time import time

#
# Configs
#

from dataclasses import dataclass
@dataclass
class ImageEncoderConfig:
    # Use a Ladder CNN in addition to the image encoder?
    use_cnn: bool=False
    # Run the resizing operations after the CNN? (might preserve fine pixel-level details better)
    resize_before_cnn: bool=True

    # use LoRa on the image encoder transformer?
    use_encoder_lora: bool=False
    encoder_lora_r: int=4

    # train the image encoder's patch embedding CNN?
    use_patch_embed: bool=False

@dataclass
class MaskDecoderConfig:
    ft: bool=False

    # Add LoRA weights to the mask decoder?
    use_decoder_lora: bool=False
    decoder_lora_r: int=4

    use_cls: bool=True

    custom_hypers: bool=True

#
# Wrappers
#

from .ladder_cnn import CNN_SAM
from .samed import LoRA_Tiny_Image_Encoder,LoRA_Mask_Decoder
from .sam_hq_loss_mask import loss_masks

from .cfg import Config

class WrappedImageEncoder(nn.Module):
    def __init__(self,
                 predictor: SamPredictor,

                cfg: Config,
    ):
        super().__init__()
        self.predictor = predictor
        self.image_encoder = predictor.model.image_encoder

        self.cfg = cfg
        self.encoder_cfg = cfg.model.encoder

        self.use_cnn = self.encoder_cfg.use_cnn
        self.use_lora = self.encoder_cfg.use_encoder_lora
        self.use_patch_embed = self.encoder_cfg.use_patch_embed

        if self.use_cnn:
            self.cnn = CNN_SAM(resize_before_cnn=self.encoder_cfg.resize_before_cnn)
        
        if self.use_lora:
            assert cfg.model.size == "vit_t", "LoRA only works with MobileSAM for now"
            self.lora = LoRA_Tiny_Image_Encoder(self.image_encoder, r=self.encoder_cfg.encoder_lora_r)
        
        self.can_cache_embeddings = cfg.train.cache_embeddings and not self.use_lora
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
                # print("Loading from cache")
                return torch.load(f"{self.cache_dir}/{hash}.pt")
        
        input_image, input_image_torch, input_image_resized = imgs

        input_image_final = self.predictor.model.preprocess(input_image_resized)

        # use no_grad if can_cache_embeddings is true
        if self.can_cache_embeddings:
            with torch.no_grad():
                features = self.image_encoder(input_image_final)[0]
        else:
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

    def get_decoder_input(self, imgs, prompt):
        # populate dict with embeddings and, possibly, ctx_embeddings.
        # this dict will get passed in as kwargs to the mask decoder.

        embeddings = self(imgs)

        ret = {
            "image_embeddings": embeddings,
            "context_embeddings": None
        }

        if prompt.context is not None:
            contexts = []
            for context in prompt.context:
                img = context.img
                mask = context.mask

                torch_mask = self.predictor.preprocess_mask(mask)
                torch_mask = F.interpolate(torch_mask, size=(256,256), mode="bilinear", align_corners=False)

                ctx_embedding = self(img)

                # mark as a "context" image
                ctx_embedding += self.predictor.model.prompt_encoder.context_embed

                # encode with prompt encoder
                _, ctx_dense_embeddings = self.predictor.model.prompt_encoder(
                    masks=mask[None,None,...],
                )
                ctx_embedding += ctx_dense_embeddings
                
                contexts.append(ctx_embedding)
            
            ret["context_embeddings"] = context_torch = torch.stack(contexts,dim=0)
            assert len(context_torch.shape) == 4,f"contexts shape is {context_torch.shape}"
        
        return ret

    def get_trainable_parameters(self):
        combined_params = []
        if self.use_cnn:
            combined_params += list(self.cnn.parameters())
        if self.use_lora:
            combined_params += list(self.lora.get_trainable_parameters())
        if self.use_patch_embed:
            combined_params += list(self.image_encoder.patch_embed.parameters())
        
        return combined_params
    
    def get_trainable_state_dict(self):
        dicts = {}
        if self.use_cnn:
            dicts["cnn"] = self.cnn.state_dict()
        if self.use_lora:
            dicts["lora"] = self.lora.get_trainable_state_dict()
        if self.use_patch_embed:
            dicts["image_encoder.patch_embed"] = self.image_encoder.patch_embed.state_dict()
        
        return flatten_state_dict(dicts)


from .binary_mask import binarize_dynamic,get_max_iou_masks
from persam.persam_f import calculate_sigmoid_focal_loss,calculate_dice_loss

class WrappedMaskDecoder(nn.Module):
    def __init__(
            self,
            predictor: SamPredictor,

            cfg: Config,
    ):
        super().__init__()
        self.predictor = predictor
        self.mask_decoder = predictor.model.mask_decoder

        self.cfg = cfg
        self.decoder_cfg = cfg.model.decoder

        self.ft = self.decoder_cfg.ft
        self.use_lora = self.decoder_cfg.use_decoder_lora
        self.use_cls = self.decoder_cfg.use_cls

        self.mask_decoder = predictor.model.mask_decoder

        if self.use_lora:
            self.lora_mask_decoder = LoRA_Mask_Decoder(self.mask_decoder, r=self.decoder_cfg.decoder_lora_r)
        
        if self.use_cls:
            num_classes = cfg.data.num_classes
            assert num_classes is not None, "Must set num_classes before initializing the mask decoder"
            self.mask_decoder.add_cls_token(num_classes, cfg.model.decoder.custom_hypers, cfg.train.only_cls_loss)

            if self.cfg.train.warm_start and cfg.model.decoder.custom_hypers:
                # warm-start the cls tokens to match the single-mask token
                self.mask_decoder.cls_mask_tokens.weight.data = self.mask_decoder.mask_tokens.weight.data[0].unsqueeze(0).repeat(num_classes,1)
                # ditto for hypernetwork MLPs
                main_hypernetwork_mlp = self.mask_decoder.output_hypernetworks_mlps[0]
                for cls_hypernetwork_mlp in self.mask_decoder.cls_hypernetworks_mlps:
                    # deep copy state dict of main hypernetwork MLP
                    cls_hypernetwork_mlp.load_state_dict(main_hypernetwork_mlp.state_dict().copy())
                print("warm started")

    
    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)
    
    # note: this is included in postprocess because it involves control flow (dynamic resizing, argmaxing+dynamic indexing).
    def postprocess(self, low_res_masks, iou_predictions, sizes):
        original_size,input_size = sizes

        upscaled_masks = self.predictor.model.postprocess_masks(low_res_masks,input_size,original_size).squeeze(0)

        should_binarize_dynamic = self.cfg.model.binarize_dynamic == "true" or (not self.training and self.cfg.model.binarize_dynamic == "eval")
        binary_masks = binarize_dynamic(upscaled_masks) if should_binarize_dynamic else upscaled_masks > 0

        max_idx = torch.argmax(iou_predictions)

        return \
            (upscaled_masks,binary_masks), max_idx
        
    def calculate_mask_loss(self,pred_masks,upscaled_masks,gt_masks,losses):
        num_masks = pred_masks.shape[0]
        assert num_masks == gt_masks.shape[0]
        # print("pred_masks",pred_masks.shape,"upscaled_masks",upscaled_masks.shape,"gt_masks",gt_masks.shape)
        if self.cfg.train.sam_hq_loss:
            focal,dice = loss_masks(pred_masks,gt_masks,num_masks)
            losses["focal"] = focal
            losses["dice"] = dice

        else:
            flat_pred_mask = upscaled_masks.view(num_masks,-1)
            flat_gt_mask = gt_masks.view(num_masks,-1)

            losses["focal"] = calculate_sigmoid_focal_loss(flat_pred_mask,flat_gt_mask,should_sigmoid=True)
            losses["dice"] = calculate_dice_loss(flat_pred_mask,flat_gt_mask,should_sigmoid=True)

        losses["mask"] = losses["focal"] + losses["dice"]

    
    # TODO run focal and dice loss on batch of masks, rather than running on each mask and averaging
    def loss(self,
             low_res_masks,iou_predictions, cls_low_res_masks,cls_iou_predictions,
             gt_info,gt_cls_info, sizes,prompt
            ):

            def add_losses(losses):
                loss = torch.tensor(0,dtype=torch.float32,device=gt_info["masks"].device)
                for k,v in losses.items():
                    scale = self.cfg.train.loss_scales.get(k,0)
                    if scale != 0:
                        # print(f"Adding {k} loss with scale {scale}, value {v}")
                        loss += scale*v
                losses["loss"] = loss

            use_cls_loss = self.use_cls and gt_cls_info is not None
            use_normal_loss = not (self.cfg.train.only_cls_loss and use_cls_loss)

            calculate_normal_loss = not (self.cfg.train.only_cls_loss and use_cls_loss and self.training)

            losses = {}

            gt_masks = gt_info["masks"]

            # normal loss
            if calculate_normal_loss:
                assert low_res_masks is not None,"low_res_masks is None"
                assert iou_predictions is not None,"iou_predictions is None"

                (upscaled_masks,binary_masks), max_idx = self.postprocess(low_res_masks,iou_predictions, sizes)

                gt_binary_mask, _, max_iou, best_pred_idx,_ = get_max_iou_masks(gt_masks,binary_masks[None,max_idx,...])

                # mse loss
                losses["mse"] = F.mse_loss(max_iou, iou_predictions[0,best_pred_idx])

                if prompt.mask_loss and self.cfg.data.use_masks:
                    pred_mask = low_res_masks[:,best_pred_idx,None]
                    # print("upscaled_masks",upscaled_masks.shape)
                    upscaled_mask = upscaled_masks[best_pred_idx,None]
                    self.calculate_mask_loss(pred_mask,upscaled_mask,gt_binary_mask[None,None],losses)

                add_losses(losses)
            
            # cls loss
            if use_cls_loss:
                before_cls = time()
                cls_losses = {}

                gt_cls = gt_cls_info["gt_cls"]
                gt_cls_logits = gt_cls_info["gt_cls_one_hot"]

                before_postprocess = time()
                (cls_upscaled_masks,cls_binary_masks), max_cls_idx = self.postprocess(cls_low_res_masks,cls_iou_predictions, sizes)

                before_max_iou = time()
                inputs = [gt_masks,cls_binary_masks,gt_cls,torch.arange(self.cfg.data.num_classes,device=gt_masks.device)]
                cls_gt_binary_mask, cls_binary_mask, cls_max_iou, best_cls, best_det = get_max_iou_masks(*inputs)
                cls_pred_iou = F.sigmoid(cls_iou_predictions[0,best_cls])

                assert best_cls == gt_cls[best_det], f"best_cls is {best_cls} but gt_cls is {gt_cls[best_det]}"

                # simultaneously treat cls iou predictions as probabilities and IoU logits.
                # I think this is OK because cross-entropy loss is bias-independent.
                before_logit_losses = time()
                cls_losses["ce"] = F.cross_entropy(cls_iou_predictions[0],gt_cls_logits[best_det])
                cls_losses["mse"] = F.mse_loss(cls_max_iou, cls_pred_iou)

                if prompt.mask_loss and self.cfg.data.use_masks:
                    before_mask_loss = time()

                    cls_pred_mask = cls_low_res_masks[:,best_cls,None]
                    # print("cls_upscaled_masks",cls_upscaled_masks.shape)
                    cls_upscaled_mask = cls_upscaled_masks[best_cls,None]
                    self.calculate_mask_loss(cls_pred_mask,cls_upscaled_mask,cls_gt_binary_mask[None,None],cls_losses)

                
                add_losses(cls_losses)

                for k,v in cls_losses.items():
                    losses[f"cls_{k}"] = v
                
                
            loss = torch.tensor(0,dtype=torch.float32,device=gt_masks.device)
            if use_normal_loss:
                loss += losses["loss"]
            if use_cls_loss:
                loss += losses["cls_loss"]

            return loss,losses

    def get_trainable_parameters(self):
        if self.ft:
            return list(self.mask_decoder.parameters())

        combined_params = []
        if self.use_lora:
            combined_params += list(self.lora_mask_decoder.get_parameters())
        if self.use_cls:
            for param_set in [self.mask_decoder.cls_mask_tokens,self.mask_decoder.cls_iou_token,self.mask_decoder.cls_hypernetworks_mlps,self.mask_decoder.cls_iou_prediction_head]:
                if param_set is not None:
                    combined_params += list(param_set.parameters())
        
        return combined_params
    
    def get_trainable_state_dict(self):
        dicts = {}
        if self.ft:
            dicts["mask_decoder"] = self.mask_decoder.state_dict()
        else:
            if self.use_lora:
                dicts["lora_mask_decoder"] = self.lora_mask_decoder.get_trainable_state_dict()
            if self.use_cls:
                dicts["mask_decoder.cls_mask_tokens"] = self.mask_decoder.cls_mask_tokens.state_dict()
                dicts["mask_decoder.cls_iou_token"] = self.mask_decoder.cls_iou_token.state_dict()
                dicts["mask_decoder.cls_hypernetworks_mlps"] = self.mask_decoder.cls_hypernetworks_mlps.state_dict()
                dicts["mask_decoder.cls_iou_prediction_head"] = self.mask_decoder.cls_iou_prediction_head.state_dict()
        
        return flatten_state_dict(dicts)

from persam.load import load_predictor
class WrappedSamModel(nn.Module):
    def __init__(self,
                 cfg: Config,
                 predictor: SamPredictor=None,
                 ):
        super().__init__()
        self.predictor = load_predictor(cfg.model.size)
        self.predictor.model.prompt_encoder.cpu()

        self.decoder = WrappedMaskDecoder(self.predictor,cfg)
        self.encoder = WrappedImageEncoder(self.predictor,cfg)

        self.cfg = cfg
    
    def get_trainable_parameters(self):
        return self.decoder.get_trainable_parameters() + self.encoder.get_trainable_parameters()
    
    def get_trainable_state_dict(self):
        return flatten_state_dict({
            "encoder": self.encoder.get_trainable_state_dict(),
            "decoder": self.decoder.get_trainable_state_dict()
        })
    
def flatten_state_dict(dict:dict)->dict:
    ret = {}
    for k,v in dict.items():
        for k2,v2 in v.items():
            ret[f"{k}.{k2}"] = v2
    return ret