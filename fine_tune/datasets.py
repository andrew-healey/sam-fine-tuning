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
