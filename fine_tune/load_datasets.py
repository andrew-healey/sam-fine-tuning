from .cfg import Config,DataConfig
from typing import Tuple

from supervision import DetectionDataset
import supervision as sv
import os

from .datasets import extract_classes_from_dataset,shrink_dataset_to_size
from .common import grow_dataset_masks

from roboflow.core.dataset import Dataset
from typing import Union,Optional


def check_for_overlap(train_dataset:DetectionDataset,valid_dataset:DetectionDataset):
    valid_names = set(k.split(".rf")[0] for k in valid_dataset.images.keys())
    train_names = set(k.split(".rf")[0] for k in train_dataset.images.keys())

    # Check that there's no training/valid pollution
    assert len(valid_names.intersection(train_names)) == 0,"There is overlap between the training and validation sets."

def load_datasets(cfg:DataConfig, rf_dataset:Union[Dataset,str]) -> Tuple[DetectionDataset,DetectionDataset]:

    dataset_location = rf_dataset.location if isinstance(rf_dataset,Dataset) else rf_dataset
    assert type(dataset_location) == str,f"dataset_location: {dataset_location}, type: {type(dataset_location)}"

    cfg.dataset_name = os.path.basename(dataset_location)

    train_dataset = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset_location}/train",
        annotations_path=f"{dataset_location}/train/_annotations.coco.json",
        force_masks=cfg.use_masks
    )

    if cfg.create_valid:
        # split the training set into train/valid
        train_dataset,valid_dataset = train_dataset.split(0.8)

    if cfg.train_size is not None:
        train_dataset = shrink_dataset_to_size(train_dataset,cfg.train_size)
    
    if cfg.cls_ids is not None:
        print("Selecting classes",[train_dataset.classes[i] for i in cfg.cls_ids])
        train_dataset = extract_classes_from_dataset(train_dataset,cfg.cls_ids)

    if not cfg.create_valid:
        valid_dataset = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_location}/valid",
            annotations_path=f"{dataset_location}/valid/_annotations.coco.json",
            force_masks=cfg.use_masks
        )

    if cfg.valid_size is not None:
        valid_dataset = shrink_dataset_to_size(valid_dataset,cfg.valid_size)

    if cfg.cls_ids is not None:
        valid_dataset = extract_classes_from_dataset(valid_dataset,cfg.cls_ids)
    
    # I grow masks in-place for speed
    if cfg.grow_masks:
        for dataset in [train_dataset,valid_dataset]:
            grow_dataset_masks(dataset,growth_radius=cfg.data.growth_radius)

    check_for_overlap(train_dataset,valid_dataset)

    cfg.num_classes = len(train_dataset.classes)

    return train_dataset,valid_dataset

from .common import SamDataset,SplitSamDataset,SamComboDataset,RandomPointDataset,SamSemSegDataset,SamBoxDataset,SamPointDataset,SamDummyMaskDataset,SamEverythingDataset
sam_dataset_registry = {
    "sem_seg": lambda ds,cfg,args: SamSemSegDataset(ds,*args),
    "box": lambda ds,cfg,args: SamBoxDataset(ds,*args),
    "point": lambda ds,cfg,args: SamPointDataset(ds,*args,points_per_mask=cfg.data.points_per_mask),
    "dummy": lambda ds,cfg,args: SamDummyMaskDataset(ds,*args),
    "everything": lambda ds,cfg,args: SamEverythingDataset(ds,*args,points_per_side=cfg.data.points_per_side,top_k=None),
    "random": lambda ds,cfg,args: RandomPointDataset(ds,*args,points_per_img=cfg.data.points_per_img),
}

from segment_anything import SamPredictor
import numpy as np

def prepare_torch_dataset(predictor:SamPredictor,cfg:Config,ds:Dataset,max_prompts:Optional[int]=None)->sv.DetectionDataset:
    args = [predictor]

    datasets = []
    for task in cfg.data.tasks:
        datasets.append(sam_dataset_registry[task](ds,cfg,args))
    ret = SamComboDataset(datasets,*args)

    num_prompts = len(ret)
    if max_prompts is not None and num_prompts > max_prompts:
        # split off prompts
        ret = SplitSamDataset(ret,max_prompts/num_prompts)
    return ret

from src.utils.cloud_utils import firestore,gac_json,gcp_download
import os
import json

import requests
from PIL import Image

# for async download
import asyncio

async def download_img(img_id,owner_id,save_path):
    # url = "https://storage.googleapis.com/roboflow-platform-sources/{image['owner']}/{image['id']}/thumb.jpg"
    url = f"https://storage.googleapis.com/roboflow-staging-sources/{owner_id}/{img_id}/original.jpg"

    # retry 5 times
    for i in range(5):
        try:
            r = requests.get(url)
            break
        except:
            print(f"Failed to download {url}, retrying...")
    
    with open(save_path,"wb") as f:
        f.write(r.content)

    # return height,width
    return Image.open(save_path).size
    

import datetime

def download_raw_dataset(dataset_id:str,save_dir="dataset"):
    db = firestore.Client.from_service_account_json(gac_json)


    # delete the directory if it exists
    if os.path.exists(save_dir):
        os.system(f"rm -rf {save_dir}")
    os.mkdir(save_dir)

    dataset = db.collection("datasets").document(dataset_id).get().to_dict()
    annot_name = dataset["annotation"]

    sorted_classes = sorted(dataset["classes"].keys())
    inverse_idx = {k:i for i,k in enumerate(sorted_classes)}

    # mkdir train, valid, and test

    cocos = { }
    for split in ["train","valid","test"]:
        os.mkdir(f"{save_dir}/{split}")

        cocos[split] = {
            "categories":[
                {
                    "id": i,
                    "name": name,
                    "supercategory": "none"
                } for i,name in enumerate(sorted_classes)
            ],
            "images": [],
            "annotations": [],
        }

    sources = db.collection("sources").where("projects","array_contains",dataset_id).get()

    # sort by update date - "updated" is a firestore timestamp
    sources = sorted(sources,key=lambda x:x.to_dict()["updated"])

    for source in sources:
        annot = source.to_dict()
        split = annot.get(f"split.{dataset_id}",None)
        if not split:
            continue

        i = len(cocos[split]["images"])
        img_id = source.id
        owner_id = annot["owner"]
        name = annot["name"]

        # download image
        img_path = f"{save_dir}/{split}/{name}"

        # run in the background:
        asyncio.run(download_img(img_id,owner_id,img_path))

        updated_iso = annot["updated"].isoformat()

        images_entry = {
            "id":i,
            "file_name":name,
            "license":None,
            "height":annot["height"],
            "width":annot["width"],
            "date_captured":updated_iso,
        }

        cocos[split]["images"].append(images_entry)

        # add annotations
        converted = json.loads(annot["annotations"][annot_name]["converted"])
        key = converted["key"]
        boxes = converted["boxes"]

        for box in boxes:
            coco_annot = {}
            coco_annot["bbox"] = [box["x"],box["y"],box["width"],box["height"]]
            coco_annot["bbox"] = [float(x) for x in coco_annot["bbox"]]

            if "points" in box:
                # flatten 2d array to 1d
                coco_annot["segmentation"] = [x for y in box["points"] for x in y]
            
            coco_annot["iscrowd"] = 0
            coco_annot["image_id"] = i
            coco_annot["category_id"] = inverse_idx[box["label"]]
            coco_annot["id"] = len(cocos[split]["annotations"])

            cocos[split]["annotations"].append(coco_annot)
            
    for split in ["train","valid","test"]:
        with open(f"{save_dir}/{split}/_annotations.coco.json","w") as f:
            json.dump(cocos[split],f)
    
    print(f"Loaded {len(sources)} images from dataset {dataset_id} into {save_dir}")