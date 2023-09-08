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
- Clicks-per-instance benchmarking - DONE!
- Confusion matrix benchmarking - DONE!
- Unified benchmarking
- Automatically pick best pred-IoU threshold for SAM vs. custom SAM
- Attn masks + duplicate points: LHS can't attend to RHS, vice versa - or maybe pad + batch - DONE!
- Simple Flask server for requesting+downloading a trained model
- ONNX export as a function - DONE!
- Train loop as a function - no custom configs yet, just set defaults
- Model-ify the configurable encoder & decoder - DONE!
"""

import sys
import os
sys.path.insert(0, os.path.relpath("./segment-anything"))
import segment_anything

# add ../roboflow-train/train/ to the path (relative to this file)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),"..","roboflow-train","train"))
from src.trainer import Trainer

from .viz import show_confusion_matrix,plt_to_pil,clip_together_imgs,mask_to_img # configure headless matplotlib
from .models import ImageEncoderConfig,MaskDecoderConfig,WrappedSamModel
from .cfg import Config,DataConfig,ModelConfig,TrainConfig
from .load_datasets import load_datasets,prepare_torch_dataset,download_raw_dataset
from .datasets import get_class_counts
from .common import SamDataset,get_max_iou_masks
from .optimizer import get_optimizer


import numpy as np

from typing import List
from dataclasses import asdict

import wandb
wandb.login()

from tqdm import tqdm
from numpy.random import permutation
import torch
from random import randrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomSAMTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.cfg = self.make_config()
        self.load_datasets()

        self.sam = WrappedSamModel(self.cfg).to(device)
    
    def load_datasets(self):

        # download from roboflow
        download_raw_dataset(self.dataset_id,self.dataset_dir)

        self.sv_train_dataset,self.sv_valid_dataset = load_datasets(self.cfg,self.dataset_dir)

        self.train_dataset = prepare_torch_dataset(self.predictor,self.cfg,self.sv_train_dataset,max_prompts=self.cfg.data.train_prompts)
        self.valid_dataset = prepare_torch_dataset(self.predictor,self.cfg,self.sv_valid_dataset,max_prompts=self.cfg.data.valid_prompts)

        self.cls_counts = get_class_counts(self.sv_train_dataset)
    
    def get_points_per_mask(self)->List[int]:
        # try to fix training imbalances

        min_prompts_per_cls = 100
        max_points_per_mask = 10
        
        # for all classes with < min_prompts_per_cls, give them a multiplier to get them up to min_prompts_per_cls
        cls_multipliers = np.ones(len(self.sv_train_dataset.classes))
        for cls,count in enumerate(self.cls_counts):
            if count < min_prompts_per_cls and count > 0:
                cls_multipliers[cls] = min(max_points_per_mask,min_prompts_per_cls // count)
        
        return cls_multipliers.tolist()
    
    def make_config(self)->Config:

        # TODO choose whether to use lora & patch embeddings based on # of instances in dataset

        return Config(
            data=DataConfig(
                tasks=["point","box"],
                train_prompts=10_000,
                valid_prompts=200,
                points_per_mask=self.get_points_per_mask(),
            ),
            model=ModelConfig(
                size="vit_t",
                encoder=ImageEncoderConfig(
                    use_patch_embed=False
                ),
                decoder=MaskDecoderConfig(
                    use_lora=False,
                ),
            ),
            train=TrainConfig(
                initial_lr=2e-4,
                cache_embeddings=True,
                run_grad=True,
                export_full_decoder=True,
            )
        )
    
    def _train(self):

        cfg = self.cfg

        run = wandb.init(
            project="sam-fine-tune-prod",
            config=asdict(cfg)
        )

        optimizer,scheduler = get_optimizer(cfg,self.sam)

        curr_iters = 0
        accumulated_loss = 0

        # track running avg of loss
        recent_losses = []

        curr_epoch = 0

        # iter through dataset in random order
        while curr_iters < cfg.train.max_steps:
            self.evaluate()
            for i,idx in enumerate(tqdm(permutation(len(self.train_dataset)))):

                with torch.no_grad():
                    prompt_input, gt_info,gt_cls_info, imgs,sizes, prompt = batch = SamDataset.to_device(self.train_dataset[idx],device)
                
                encoder_output = self.sam.encoder.get_decoder_input(imgs,prompt)
                pred = self.sam.decoder(**prompt_input,**encoder_output)

                #
                # WandB
                #
                
                loss_dict = self.sam.decoder.loss(*pred, gt_info,gt_cls_info, sizes,prompt)
                loss = loss_dict["loss"]


                loss_dict = {k:v.item() for k,v in loss_dict.items()}
                wandb.log(loss_dict)

                #
                # Logging
                #

                recent_losses += [loss.item()]
                recent_losses = recent_losses[-cfg.train.log_period:]

                if curr_iters % cfg.train.eval_period == 0:
                    pass

                if curr_iters % cfg.train.log_period == 0:
                    print(f"Loss: {sum(recent_losses)/len(recent_losses)}")

                curr_iters += 1

                if not cfg.train.run_grad: continue

                accumulated_loss += loss
                if curr_iters % cfg.train.batch_size == 0:
                    optimizer.zero_grad()
                    accumulated_loss /= torch.tensor(cfg.train.batch_size,dtype=torch.float32)
                    accumulated_loss.backward()
                    optimizer.step()
                    accumulated_loss = 0
                
                scheduler.step()

            curr_epoch += 1
    
    def evaluate(self):
        cfg = self.cfg
        # make confusion matrix, compute loss

        pred_classes = []
        gt_classes = []

        running_loss = 0.0
        running_count = 0

        viz_idx = randrange(len(self.valid_dataset))

        viz_gt_mask = viz_cls_mask = viz_mask = viz_img = None

        for batch in tqdm(self.valid_dataset):

            batch = SamDataset.to_device(batch,device)
            prompt_input, gt_info, gt_cls_info, imgs,sizes, prompt = batch

            use_cls = cfg.model.decoder.use_cls and gt_cls_info is not None
            assert use_cls,"This function only works for cls models rn."

            with torch.no_grad():
                encoder_output = self.sam.encoder.get_decoder_input(imgs,prompt)

                low_res_masks, iou_predictions, cls_low_res_masks,cls_iou_predictions = pred = self.sam.decoder(**prompt_input,**encoder_output)

                losses = self.sam.decoder.loss(*pred, gt_info,gt_cls_info, sizes,prompt)
                loss = losses["loss"]

                # get pred gt class
                (_,binary_masks), max_idx = self.sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)
                gt_binary_mask,*_ = get_max_iou_masks(gt_info["masks"],binary_masks[None,max_idx,...])

                if use_cls:
                    # get pred gt class
                    (_,cls_binary_masks), pred_cls = self.sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)

                    cls_gt_binary_mask,_,_,best_cls,_ = get_max_iou_masks(gt_info["masks"],cls_binary_masks,gt_cls_info["gt_cls"],torch.arange(cfg.data.num_classes).to(device))

                    if viz_idx == running_count:
                        viz_img = imgs[0]
                        viz_gt_mask = gt_binary_mask
                        viz_cls_mask = cls_binary_masks[pred_cls,...]
                        viz_mask = binary_masks[max_idx,...]

                    pred_classes.append(pred_cls)
                    gt_classes.append(best_cls)

                running_loss += loss.item()
                running_count += 1
        valid_loss = running_loss/running_count

        assert viz_gt_mask is not None,"viz_gt_mask is None"
        assert viz_cls_mask is not None,"viz_cls_mask is None"
        assert viz_mask is not None,"viz_mask is None"

        wandb.log({
            "valid_loss": valid_loss,
            "normal_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),
            "cls_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_cls_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),
        })

        print(f"VALID - Loss: {valid_loss:.4f}")

        assert len(gt_classes) > 0,"No gt classes found"

        # calculate confusion matrix
        conf_matrix = show_confusion_matrix(gt_classes, pred_classes, class_names=self.valid_dataset.classes)
        percent_recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 1)
        avg_recall = np.mean(percent_recall)

        confusion_matrix = plt_to_pil()
        wandb.log({
            "confusion_matrix": wandb.Image(confusion_matrix),
            "avg_recall": avg_recall.item(),
        })