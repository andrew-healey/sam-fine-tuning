import supervision as sv
import numpy as np
from supervision import DetectionDataset

from typing import List

# This acts like an oracle for a given FewShotOntology.
def extract_classes_from_dataset(
    old_dataset: DetectionDataset, class_ids: List[int]
) -> DetectionDataset:
    """
    Extract a subset of classes from a dataset.
    This re-maps the class_ids to be contiguous.

    Keyword arguments:
    old_dataset -- the dataset to extract from
    class_ids -- the class_ids (as integers) to extract
    """

    new_annotations = {}
    for img_name, detections in old_dataset.annotations.items():
        new_detectionss = []
        for new_class_id, class_id in enumerate(class_ids):
            new_detections = detections[detections.class_id == class_id]
            new_detections.class_id = (
                np.ones_like(new_detections.class_id) * new_class_id
            )

            new_detectionss.append(new_detections)
        new_annotations[img_name] = sv.Detections.merge(new_detectionss)

    classes = [old_dataset.classes[class_id] for class_id in class_ids]
    return sv.DetectionDataset(
        classes=classes, images=old_dataset.images, annotations=new_annotations
    )

from random import sample
def shrink_dataset_to_size(
    dataset: DetectionDataset, max_imgs: int = 15
) -> DetectionDataset:
    """
    Pick a random subset of the dataset.

    Keyword arguments:
    dataset -- the dataset to shrink
    max_imgs -- the maximum number of images to keep in the dataset
    """
    imgs = list(dataset.images.keys())

    if len(imgs) <= max_imgs:
        # copy dataset
        return DetectionDataset(
            classes=dataset.classes,
            images={**dataset.images},
            annotations={**dataset.annotations},
        )

    imgs = sample(imgs, max_imgs)

    new_images = {img_name: dataset.images[img_name] for img_name in imgs}
    new_annotations = {img_name: dataset.annotations[img_name] for img_name in imgs}

    return DetectionDataset(
        classes=dataset.classes, images=new_images, annotations=new_annotations
    )

from torch.utils.data import Dataset

def save_torch_dataset(name: str, dataset: Dataset):
    folder = os.path.join("data",name)
    os.rmdir(folder, ignore_errors=True)
    os.mkdirs(folder)

    for i, entry in enumerate(dataset):
        torch.save(entry, os.path.join(folder, f"{i}.pt"))

from glob import glob

class CachedDataset(Dataset):
    def __init__(self, files: List[str]):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def load_torch_datasets() -> Dataset:
    files = glob("data/**/*.pt")
    return CachedDataset(files)

def merge_datasets(
    old_dataset: DetectionDataset, new_dataset: DetectionDataset,overwrite=False
)-> DetectionDataset:

    if old_dataset.classes != new_dataset.classes:
        new_classes = [*old_dataset.classes,*new_dataset.classes]
        # switch class_ids in new_dataset to match old_dataset
        new_annotations = {}
        for img_name, detections in new_dataset.annotations.items():
            new_detections = sv.Detections.merge([detections])
            new_detections.class_id += len(old_dataset.classes)
            new_annotations[img_name] = new_detections
    else:
        new_classes = old_dataset.classes
        new_annotations = new_dataset.annotations
    
    # now merge annotations for matching images

    old_annotations = {**old_dataset.annotations}

    for img_name, detections in new_annotations.items():
        if img_name in old_annotations and not overwrite:
            old_annotations[img_name] = sv.Detections.merge([old_annotations[img_name],detections])
        else:
            old_annotations[img_name] = detections
    
    return DetectionDataset(
        classes=new_classes,
        images={**old_dataset.images,**new_dataset.images},
        annotations=old_annotations,
    )

def merge_many_datasets(datasets: List[DetectionDataset], overwrite=True):
    merged_dataset = DetectionDataset(classes=[],images={},annotations={})
    for dataset in datasets:
        merged_dataset = merge_datasets(merged_dataset,dataset,overwrite=overwrite)
    return merged_dataset

import os,cv2
def load_dir_as_dataset(dir: str) -> DetectionDataset:
    extensions = ["jpg","png","jpeg"]
    files = [file for ext in extensions for file in glob(os.path.join(dir,f"*.{ext}"))]

    images = {file: cv2.imread(file) for file in files}
    annotations = {file: sv.Detections.empty() for file in files}
    classes = []

    return DetectionDataset(classes=classes,images=images,annotations=annotations)