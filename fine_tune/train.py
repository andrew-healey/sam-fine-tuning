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
from .cfg import Config,DataConfig,ModelConfig,TrainConfig,MaskDecoderConfig,ImageEncoderConfig,WandbConfig
from .load_datasets import load_datasets,prepare_torch_dataset,download_raw_dataset
from .datasets import get_class_counts
from .common import SamDataset,to
from .binary_mask import get_max_iou_masks
from .optimizer import get_optimizer
from .interaction import get_ious_and_clicks
from .export import export
from .interaction import get_next_interaction


import numpy as np

from typing import List
from dataclasses import asdict
from transformers import HfArgumentParser

import wandb

from tqdm import tqdm
from numpy.random import permutation
import torch
from random import randrange,choice
from glob import glob

import matplotlib.pyplot as plt


# wandb stuff

class CustomSAMTrainer(AbstractMonitoredTrainer):
    def __init__(self,*args,
        cache_path: str = CACHE_PATH,
        dataset_id: str = DATASET_ID,
                 **kwargs):
        super().__init__(*args,**kwargs)

        self.cache_path = cache_path
        self.model_path = os.path.join(cache_path,"model")

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.dataset_id = dataset_id
        self.dataset_dir = os.path.join(CACHE_PATH, "dataset")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert dataset_id is not None,"dataset_id is None"

        self.update_status("loading")

        self.cfg = self.make_config()
        self.load_datasets()

        self.sam = WrappedSamModel(self.cfg).to(self.device)

        self.load_torch_datasets()
    
    def load_datasets(self):

        # download from roboflow
        print(f"Downloading dataset {self.dataset_id}...")
        download_raw_dataset(self.dataset_id,self.dataset_dir)

        self.sv_train_dataset,self.sv_valid_dataset = load_datasets(self.cfg.data,self.dataset_dir)

        self.cls_counts = get_class_counts(self.sv_train_dataset)
        self.cfg.data.points_per_mask = self.get_points_per_mask()
    
    def load_torch_datasets(self):

        assert type(self.cfg.data.points_per_mask) == list,"points_per_mask is not set"

        self.train_dataset = prepare_torch_dataset(self.sam.predictor,self.cfg,self.sv_train_dataset,max_prompts=self.cfg.data.train_prompts)
        self.valid_dataset = prepare_torch_dataset(self.sam.predictor,self.cfg,self.sv_valid_dataset,max_prompts=self.cfg.data.valid_prompts)
    
    def get_points_per_mask(self)->List[int]:
        # try to fix training imbalances

        min_prompts_per_cls = 500
        max_points_per_mask = 10
        
        # for all classes with < min_prompts_per_cls, give them a multiplier to get them up to min_prompts_per_cls
        cls_multipliers = np.ones(len(self.sv_train_dataset.classes),dtype=np.int32)
        for cls,count in enumerate(self.cls_counts):
            if count < min_prompts_per_cls and count > 0:
                cls_multipliers[cls] = min(max_points_per_mask,min_prompts_per_cls // count)
        
        print("cls counts",self.cls_counts)
        print("points per mask",cls_multipliers)
        
        return cls_multipliers.tolist()
    
    def make_config(self)->Config:

        # TODO choose whether to use lora & patch embeddings based on # of instances in dataset

        # load from cmd line
        parser = HfArgumentParser((Config,DataConfig,ModelConfig,TrainConfig,MaskDecoderConfig,ImageEncoderConfig,WandbConfig))

        cfg, data_cfg, model_cfg, train_cfg, mask_decoder_cfg, image_encoder_cfg, wandb_cfg = parser.parse_args_into_dataclasses()

        model_cfg.decoder = mask_decoder_cfg
        model_cfg.encoder = image_encoder_cfg

        cfg.data = data_cfg
        cfg.model = model_cfg
        cfg.train = train_cfg
        cfg.wandb = wandb_cfg

        return cfg

    def setup_wandb(self):
        cfg = self.cfg
        commit = os.environ.get("COMMIT",None)

        if commit is None:
            # get commit from git
            import subprocess
            commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

            # check if there's a local change
            if len(subprocess.check_output(["git", "status", "--porcelain"])) > 0:
                # warn that there's a local change
                print("WARNING: Local changes detected. Consider committing these changes for reproducibility.")

        wandb.login()
        self.run = wandb.init(
            project="sam-fine-tune-prod",
            config=asdict(cfg),
            tags=[commit],
            name=cfg.wandb.name,
            group=cfg.wandb.group,
        )

    
    def _train(self):

        cfg = self.cfg

        optimizer,scheduler = get_optimizer(cfg,self.sam)

        curr_iters = 0
        accumulated_loss = 0

        # track running avg of loss
        recent_losses = []

        curr_epoch = 0

        # iter through dataset in random order
        while curr_iters < cfg.train.max_steps and curr_epoch < cfg.train.max_epochs:
            # mark as eval mode
            self.sam.eval()

            self.evaluate()

            # sometimes use cls masks for interactive segmentation, sometimes not
            use_cls_choices = [False]
            if cfg.model.decoder.use_cls:
                use_cls_choices.append(True)
                if cfg.train.only_cls_loss:
                    use_cls_choices = [True]

            # mark as train mode
            self.sam.train()
            for i,idx in enumerate(tqdm(permutation(len(self.train_dataset)))):

                with torch.no_grad():
                    prompt_input, gt_info,gt_cls_info, imgs,sizes, prompt = batch = to(self.train_dataset[idx],self.device)

                use_cls = choice(use_cls_choices)
                
                for ref_step in range(cfg.train.num_refinement_steps+1):
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

                    if ref_step < cfg.train.num_refinement_steps:
                        
                        low_res_masks,iou_predictions,cls_low_res_masks,cls_iou_predictions=pred

                        if use_cls:
                            (upscaled_masks,binary_masks), max_idx = self.sam.decoder.postprocess(cls_low_res_masks,cls_iou_predictions,sizes)

                            # get the most correct cls prediction
                            gt_binary_mask, binary_mask, max_iou, best_cls, best_det = get_max_iou_masks(gt_masks,binary_masks,gt_cls_info["gt_cls"],torch.arange(sam.cfg.data.num_classes).to(device))
                        else:
                            (upscaled_masks,binary_masks), max_idx = self.sam.decoder.postprocess(low_res_masks,iou_predictions,sizes)

                            # get the most correct cls prediction
                            gt_binary_mask, binary_mask, max_iou, best_pred, best_det = get_max_iou_masks(gt_masks,binary_masks,None,None)


                        prompt = get_next_interaction(binary_mask,best_det,prompt)

                        prompt_input,gt_masks = to(self.train_dataset.prompt_to_tensors(prompt,sizes),self.device)
                        gt_info["masks"] = gt_masks

                        gt_cls_info = to(self.train_dataset.cls_to_tensors(prompt),self.device)


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

            batch = to(batch,self.device)
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
                    cls_gt_binary_mask,_,_,best_cls,_ = get_max_iou_masks(gt_info["masks"],cls_binary_masks,gt_cls_info["gt_cls"],torch.arange(cfg.data.num_classes).to(self.device))

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

        cls_miou,normal_miou =self.get_iou_vs_clicks()

        results = {
            "valid_loss": valid_loss,

            "valid_normal_mask_loss": valid_mask_loss,
            "valid_cls_mask_loss": valid_cls_mask_loss,

            "normal_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),
            "cls_viz": wandb.Image(clip_together_imgs(mask_to_img(viz_cls_mask,viz_img),mask_to_img(viz_gt_mask,viz_img))),

            "confusion_matrix": wandb.Image(confusion_matrix),
            "avg_recall": avg_recall.item(),

            "cls_miou": cls_miou,
            "normal_miou": normal_miou,
        }

        wandb.log(results)
        return results
    
    def get_iou_vs_clicks(self):
        cls_ious_and_clicks = get_ious_and_clicks(self.sam,self.valid_dataset,self.cfg.train.benchmark_clicks,use_cls=True,device=self.device)
        cls_ious = [iou for iou,_ in cls_ious_and_clicks]
        cls_clicks = [click for _,click in cls_ious_and_clicks]

        normal_ious_and_clicks = get_ious_and_clicks(self.sam,self.valid_dataset,self.cfg.train.benchmark_clicks,use_cls=False,device=self.device)
        normal_ious = [iou for iou,_ in normal_ious_and_clicks]
        normal_clicks = [click for _,click in normal_ious_and_clicks]

        def make_graph(clicks,ious,name):
            # graph as heatmap
            max_click_num = max(self.cfg.train.benchmark_clicks)

            num_y_bins = 10

            hist_edges_x = np.arange(max_click_num+1) + 0.5
            hist_edges_y = np.linspace(0,1,num_y_bins+1)
            #clear 
            plt.clf()
            # binned ious
            plt.hist2d(clicks, ious, bins=(hist_edges_x, hist_edges_y), cmap=plt.cm.Greys)
            plt.xlabel("Clicks")
            plt.ylabel("IoU")
            plt.title(f"{name} IoU vs. Clicks")
            plt.colorbar()

            iou_vs_clicks = plt_to_pil()
            return iou_vs_clicks
        
        wandb.log({
            "iou_vs_clicks": wandb.Image(clip_together_imgs(make_graph(cls_clicks,cls_ious,"Cls"),make_graph(normal_clicks,normal_ious,"Normal")))
        })

        cls_click_one_miou = np.mean([iou for iou,click in cls_ious_and_clicks if click == 1])
        normal_click_one_miou = np.mean([iou for iou,click in normal_ious_and_clicks if click == 1])

        return cls_click_one_miou,normal_click_one_miou
    
    def export(self):
        export(self.model_path,self.cfg,self.sam,self.device)

        all_cache_files = glob(os.path.join(self.model_path,"*"))

        gcp_upload(GCP_EXPORT_BUCKET, f"smart_poly_models/{self.dataset_id}/", all_cache_files)
    
    def train(self):

        self.setup_wandb()

        self.update_status("starting")
        self._train()

        self.sam.eval()
        self.update_status("evaluating")
        results = self.evaluate()

        cls_miou = results["cls_miou"]
        normal_miou = results["normal_miou"]

        # TODO figure out what specific metric to use for these ious

        if self.cfg.train.always_export or results["avg_recall"] > 0.75 and results["valid_cls_mask_loss"] < results["valid_normal_mask_loss"] and cls_miou > normal_miou:
            print("Exporting...")
            self.update_status("exporting")
            self.export()
            self.update_status("exported")
        else:
            self.update_status("not exporting")
            print("Not exporting.")
        
        print("Done!")
