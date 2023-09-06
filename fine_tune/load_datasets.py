from .cfg import DataConfig
from typing import Tuple

from supervision import DetectionDataset
import supervision as sv
import os

from .datasets import extract_classes_from_dataset,shrink_dataset_to_size
from .common import grow_dataset_masks


def check_for_overlap(train_dataset:DetectionDataset,valid_dataset:DetectionDataset):
    valid_names = set(k.split(".rf")[0] for k in valid_dataset.images.keys())
    train_names = set(k.split(".rf")[0] for k in train_dataset.images.keys())

    # Check that there's no training/valid pollution
    assert len(valid_names.intersection(train_names)) == 0,"There is overlap between the training and validation sets."

def load_datasets(cfg:DataConfig, rf_dataset) -> Tuple[DetectionDataset,DetectionDataset]:

    cfg.dataset_name = os.path.basename(rf_dataset.location)

    train_dataset = sv.DetectionDataset.from_coco(
        images_directory_path=f"{rf_dataset.location}/train",
        annotations_path=f"{rf_dataset.location}/train/_annotations.coco.json",
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
            images_directory_path=f"{rf_dataset.location}/valid",
            annotations_path=f"{rf_dataset.location}/valid/_annotations.coco.json",
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