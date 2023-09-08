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
from src.abstract_monitored_trainer import AbstractMonitoredTrainer
from src.env import (
    ENV,
    CACHE_PATH,
    DATASET_ID,
)

GCP_EXPORT_BUCKET = f"roboflow-{ENV}-models"

from src.utils.cloud_utils import gcp_upload

from .viz import show_confusion_matrix,plt_to_pil,clip_together_imgs,mask_to_img # configure headless matplotlib
from .models import ImageEncoderConfig,MaskDecoderConfig,WrappedSamModel
from .cfg import Config,DataConfig,ModelConfig,TrainConfig
from .load_datasets import load_datasets,prepare_torch_dataset,download_raw_dataset
from .datasets import get_class_counts
from .common import SamDataset,get_max_iou_masks
from .optimizer import get_optimizer
from .interaction import get_ious_at_click_benchmarks
from .export import export


import numpy as np

from typing import List
from dataclasses import asdict

import wandb

from tqdm import tqdm
from numpy.random import permutation
import torch
from random import randrange
from glob import glob

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomSAMTrainer(AbstractMonitoredTrainer):
    def __init__(self,*args,
        cache_path: str = CACHE_PATH,
        dataset_id: str = DATASET_ID,
                 **kwargs):
        super().__init__(*args,**kwargs)

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.cache_path = cache_path
        self.dataset_id = dataset_id
        self.dataset_dir = os.path.join(CACHE_PATH, "dataset")

        assert dataset_id is not None,"dataset_id is None"

        self.update_status("loading")

        self.cfg = self.make_config()
        self.load_datasets()

        self.sam = WrappedSamModel(self.cfg).to(device)

        self.load_torch_datasets()
    
    def load_datasets(self):

        # download from roboflow
        print(f"Downloading dataset {self.dataset_id}...")
        download_raw_dataset(self.dataset_id,self.dataset_dir)

        self.sv_train_dataset,self.sv_valid_dataset = load_datasets(self.cfg.data,self.dataset_dir)

        self.cls_counts = get_class_counts(self.sv_train_dataset)
        self.cfg.data.points_per_mask = self.get_points_per_mask()
    
    def load_torch_datasets(self):

        self.train_dataset = prepare_torch_dataset(self.sam.predictor,self.cfg,self.sv_train_dataset,max_prompts=self.cfg.data.train_prompts)
        self.valid_dataset = prepare_torch_dataset(self.sam.predictor,self.cfg,self.sv_valid_dataset,max_prompts=self.cfg.data.valid_prompts)
    
    def get_points_per_mask(self)->List[int]:
        # try to fix training imbalances

        min_prompts_per_cls = 100
        max_points_per_mask = 10
        
        # for all classes with < min_prompts_per_cls, give them a multiplier to get them up to min_prompts_per_cls
        cls_multipliers = np.ones(len(self.sv_train_dataset.classes),dtype=np.int32)
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
                points_per_mask=1, # Will be changed later during load_datasets
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
                max_steps=25_000,
                max_epochs=20,
            )
        )
    
    def _train(self):

        cfg = self.cfg

        wandb.login()
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
            # mark as eval mode
            self.sam.eval()

            self.evaluate()

            # mark as train mode
            self.sam.train()
            for i,idx in enumerate(tqdm(permutation(len(self.train_dataset)))):

                with torch.no_grad():
                    prompt_input, gt_info,gt_cls_info, imgs,sizes, prompt = batch = SamDataset.to_device(self.train_dataset[idx],device)
                
                encoder_output = self.sam.encoder.get_decoder_input(imgs,prompt)
                pred = self.sam.decoder(**prompt_input,**encoder_output)

                #
                # WandB
                #
                
                loss,loss_dict = self.sam.decoder.loss(*pred, gt_info,gt_cls_info, sizes,prompt)

                loss_dict = {k:v.item() for k,v in loss_dict.items()}
                wandb.log(loss_dict)

                #
                # Logging
                #

                recent_losses += [loss_dict["cls_loss"]]
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

        running_losses = {}
        running_counts = {}

        running_loss = 0
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

                _,losses = self.sam.decoder.loss(*pred, gt_info,gt_cls_info, sizes,prompt)

                for k,v in losses.items():
                    running_losses[k] = running_losses.get(k,0) + v.item()
                    running_counts[k] = running_counts.get(k,0) + 1
                
                running_loss += losses["cls_loss"].item()
                running_count += 1

                # get pred gt mask
                (_,binary_masks), max_idx = self.sam.decoder.postprocess(low_res_masks,iou_predictions,sizes)
                gt_binary_mask,*_ = get_max_iou_masks(gt_info["masks"],binary_masks[None,max_idx,...])

                if use_cls:
                    # get pred gt cls and mask
                    (_,cls_binary_masks), pred_cls = self.sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)
                    cls_gt_binary_mask,_,_,best_cls,_ = get_max_iou_masks(gt_info["masks"],cls_binary_masks,gt_cls_info["gt_cls"],torch.arange(cfg.data.num_classes).to(device))

                    if viz_idx == running_count:
                        viz_img = imgs[0]
                        viz_gt_mask = gt_binary_mask
                        viz_cls_mask = cls_binary_masks[pred_cls,...]
                        viz_mask = binary_masks[max_idx,...]

                    pred_classes.append(pred_cls)
                    gt_classes.append(best_cls)

        valid_mask_loss = running_losses["mask"]/running_counts["mask"]
        valid_cls_mask_loss = running_losses["cls_mask"]/running_counts["cls_mask"]
        valid_loss = running_loss/running_count

        assert viz_gt_mask is not None,"viz_gt_mask is None"
        assert viz_cls_mask is not None,"viz_cls_mask is None"
        assert viz_mask is not None,"viz_mask is None"

        print(f"VALID - Loss: {valid_loss}, Mask Loss: {valid_mask_loss}, Cls Mask Loss: {valid_cls_mask_loss}")

        assert len(gt_classes) > 0,"No gt classes found"

        # calculate confusion matrix
        conf_matrix = show_confusion_matrix(gt_classes, pred_classes, class_names=self.sv_valid_dataset.classes)
        percent_recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 1)
        avg_recall = np.mean(percent_recall)

        confusion_matrix = plt_to_pil()

        results = {
            "valid_loss": valid_loss,

            "valid_normal_mask_loss": valid_mask_loss,
            "valid_cls_mask_loss": valid_cls_mask_loss,

            "normal_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),
            "cls_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_cls_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),

            "confusion_matrix": wandb.Image(confusion_matrix),
            "avg_recall": avg_recall.item(),
        }

        wandb.log(results)

        return 
    
    def get_iou_vs_clicks(self):
        cls_ious_per_benchmark = get_ious_at_click_benchmarks(self.sam,self.valid_dataset,self.cfg.train.benchmark_clicks,use_cls=True)
        normal_ious_per_benchmark = get_ious_at_click_benchmarks(self.sam,self.valid_dataset,self.cfg.train.benchmark_clicks,use_cls=False)

        # graph
        plt.plot(self.cfg.train.benchmark_clicks,cls_ious_per_benchmark,label="cls")
        plt.plot(self.cfg.train.benchmark_clicks,normal_ious_per_benchmark,label="normal")
        plt.legend()
        plt.xlabel("Clicks")
        plt.ylabel("IoU")
        plt.title("IoU vs. Clicks")

        iou_vs_clicks = plt_to_pil()

        wandb.log({
            "iou_vs_clicks": wandb.Image(iou_vs_clicks),
        })

        return cls_ious_per_benchmark,normal_ious_per_benchmark
    
    def export(self):
        export(CACHE_PATH,self.cfg,self.sam)

        all_cache_files = glob(os.path.join(CACHE_PATH,"*"))

        gcp_upload(GCP_EXPORT_BUCKET, f"smart_poly_models/{self.dataset_id}/", all_cache_files)
    
    def train(self):

        self.update_status("starting")
        self._train()

        self.sam.eval()
        self.update_status("evaluating")
        results = self.evaluate()

        self.get_iou_vs_clicks()

        self.sam.predict()
        # TODO figure out what specific metric to use for these ious

        if results["avg_recall"] > 0.75 and results["valid_cls_mask_loss"] < results["valid_normal_mask_loss"]:
            print("Exporting...")
            self.update_status("exporting")
            self.export()
            self.update_status("exported")
        else:
            self.update_status("not exporting")
            print("Not exporting.")
