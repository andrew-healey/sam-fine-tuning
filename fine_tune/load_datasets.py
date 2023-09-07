from .cfg import Config,DataConfig
from typing import Tuple

from supervision import DetectionDataset
import supervision as sv
import os

from .datasets import extract_classes_from_dataset,shrink_dataset_to_size
from .common import grow_dataset_masks

from roboflow import Project
from typing import Union,Optional


def check_for_overlap(train_dataset:DetectionDataset,valid_dataset:DetectionDataset):
    valid_names = set(k.split(".rf")[0] for k in valid_dataset.images.keys())
    train_names = set(k.split(".rf")[0] for k in train_dataset.images.keys())

    # Check that there's no training/valid pollution
    assert len(valid_names.intersection(train_names)) == 0,"There is overlap between the training and validation sets."

def load_datasets(cfg:DataConfig, rf_dataset:Union[Project,str]) -> Tuple[DetectionDataset,DetectionDataset]:

    dataset_location = rf_dataset.location if isinstance(rf_dataset,Project) else rf_dataset

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

from .common import SamDataset,SamComboDataset,RandomPointDataset,SamSemSegDataset,SamBoxDataset,SamPointDataset,SamDummyMaskDataset,SamEverythingDataset
sam_dataset_registry = {
    "sem_seg": lambda ds,cfg,args: SamSemSegDataset(ds,*args),
    "box": lambda ds,cfg,args: SamBoxDataset(ds,*args),
    "point": lambda ds,cfg,args: SamPointDataset(ds,*args,points_per_mask=cfg.data.points_per_mask),
    "dummy": lambda ds,cfg,args: SamDummyMaskDataset(ds,*args),
    "everything": lambda ds,cfg,args: SamEverythingDataset(ds,*args,points_per_side=cfg.data.points_per_side,top_k=None),
    "random": lambda ds,cfg,args: RandomPointDataset(ds,*args,points_per_img=cfg.data.points_per_img),
}

from segment_anything import SamPredictor
from torch.utils.data import random_split

def prepare_torch_dataset(predictor:SamPredictor,cfg:Config,ds:Project,max_prompts:Optional[int]=None)->sv.DetectionDataset:
    args = [predictor]

    datasets = []
    for task in cfg.data.tasks:
        datasets.append(sam_dataset_registry[task](ds))
    ret = SamComboDataset(datasets,*args)

    num_prompts = len(ret)
    if max_prompts is not None and num_prompts > max_prompts:
        # split off prompts
        ret,_ = random_split(ret,[max_prompts,num_prompts-max_prompts])
    return ret

from src.utils.cloud_utils import firestore,gac_json
def download_raw_dataset(project:str):
    raise NotImplementedError("Haven't gotten Firestore stuff yet.")
    db = firestore.Client.from_service_account_json(gac_json)

    # js equivalent:
    # await db.collection(“sources”).where(“project”, “array-contains”, PROJECT_ID).limit(50).get()

    sources = db.collection("sources").where("project","array_contains",project).get()

    print(sources)

    raise 1

download_raw_dataset("climbing-z0pqv")