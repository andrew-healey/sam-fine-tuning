import supervision as sv
from supervision import Detections,MaskAnnotator
from supervision.detection.utils import mask_to_xyxy
import cv2

from PIL import Image
import numpy as np

annotator = MaskAnnotator()
def mask_to_img(mask,img):
    mask = mask.cpu().detach().numpy()[None,:,:]
    dets = Detections(
        mask=mask,
        xyxy=mask_to_xyxy(mask),
    )
    ann_img = annotator.annotate(scene=img,detections=dets)
    return Image.fromarray(ann_img).convert('RGB')

def clip_together_imgs(img1,img2):
    return Image.fromarray(np.hstack((img1,img2))).convert('RGB')

from typing import List
import matplotlib.pyplot as plt
def show_confusion_matrix(gt_classes: List[int], pred_classes: List[int], class_names: List[str]):
    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes,num_classes))
    for gt, pred in zip(gt_classes, pred_classes):
        conf_matrix[gt,pred] += 1

    plt.matshow(conf_matrix, cmap=plt.cm.Blues)
    # show numbers
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(x=j, y=i, s=int(conf_matrix[i, j]), va='center', ha='center', size='xx-large')
    
    # axes
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)
    # labels
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")

    plt.show()

box_annotator = sv.BoxAnnotator()

def render_prompt(img,prompt,sv_dataset):
    if isinstance(img,str):
        img = sv_dataset.images[img]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    gt_masks = prompt.gt_mask[None,...] if prompt.gt_mask is not None else prompt.gt_masks
    gt_clss = prompt.gt_clss if prompt.gt_clss is not None else prompt.gt_cls[None,...] if prompt.gt_cls is not None else np.zeros(len(gt_masks),dtype=int)
    gt_detection = sv.Detections(
        xyxy=sv.detection.utils.mask_to_xyxy(gt_masks),
        mask=gt_masks,
        class_id=gt_clss,
    )

    if prompt.mask is None:
        mask_detection = sv.Detections.empty()
    else:
        mask_detection = sv.Detections(
            xyxy=sv.detection.utils.mask_to_xyxy(prompt.mask[None,...]),
            mask=prompt.mask[None,...],
            class_id=np.array([1]),
        )
    
    if prompt.box is None:
        box_detection = sv.Detections.empty()
    else:
        box_detection = sv.Detections(
            xyxy=prompt.box[None,...],
            class_id=np.array([2]),
        )

    if prompt.points is None:
        point_detection = sv.Detections.empty()
    else:
        class_ids = prompt.labels + 3
        masks = np.zeros((len(prompt.points),*img.shape[:2]),dtype=bool)
        for i,point in enumerate(prompt.points):
            # draw a circle
            white_circle = np.zeros(img.shape[:2],dtype=np.uint8)
            cv2.circle(white_circle,point,10,255,-1)

            masks[i] = white_circle.astype(bool)

        point_detection = sv.Detections(
            xyxy=sv.detection.utils.mask_to_xyxy(masks),
            mask=masks,
            class_id=class_ids,
        )

    detections = sv.Detections.merge([gt_detection,mask_detection])

    shown_img = annotator.annotate(scene=img,detections=detections)
    shown_img = box_annotator.annotate(scene=shown_img,detections=box_detection)
    shown_img = annotator.annotate(scene=shown_img,detections=point_detection)

    return Image.fromarray(shown_img).convert("RGB")