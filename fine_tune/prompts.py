from numpy import ndarray
import cv2
import supervision as sv
from PIL import Image
import numpy as np

from typing import Tuple,Union,Dict,List

# Coords = Tuple[ndarray,ndarray] # shape (N, 2), dtype float32; shape (N,), dtype bool
Box = ndarray # shape (4,), dtype float32, in XYXY format
BoolMask = ndarray # shape (H,W), dtype bool
FloatMask = ndarray # shape (H,W), dtype float32, in [0,1]

# Prompt = Dict[str,Union[Coords,Box,BoolMask]]

# using a dataclass instead:
from dataclasses import dataclass,asdict

@dataclass
class ContextPair:
    img: np.ndarray
    gt_mask: np.ndarray

@dataclass
class Prompt:
    points: ndarray = None
    labels: ndarray = None
    box: Box = None
    mask: BoolMask = None # TODO: also allow FloatMask

    gt_mask: BoolMask = None
    gt_masks: BoolMask = None

    context: List[ContextPair] = None

    multimask: bool = False

    def __post_init__(self):
        # assert self.points is not None or self.box is not None or self.mask is not None,"Prompt must have at least one of coords, box, or mask"

        if self.box is not None:
            assert self.box.shape == (4,),f"box must have shape (4,), not {self.box.shape}"

        # default to all points being foreground
        if self.points is not None and self.labels is None:
            self.labels = np.ones(len(self.points),dtype=bool)
        if self.points is not None or self.labels is not None:
            assert len(self.points.shape) == 2 and self.points.shape[1] == 2,f"points must have shape (N,2), not {self.points.shape}"
            assert len(self.labels.shape) == 1,f"points must have shape (N,), not {self.points.shape}"
            assert self.labels.dtype == bool,f"labels must have dtype bool, not {self.labels.dtype}"
            assert len(self.points) == len(self.labels),f"points and labels must have the same length, not {len(self.points)} and {len(self.labels)}"

        if self.mask is not None:
            assert len(self.mask.shape) == 2,f"mask must have shape (H,W), not {self.mask.shape}"
            assert self.mask.dtype == bool,f"mask must have dtype bool, not {self.mask.dtype}"

        assert (self.gt_mask is None) != (self.gt_masks is None),"Prompt must have exactly one of gt_mask or gt_masks"


annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

def render_prompt(name,prompt,sv_dataset):
    img = sv_dataset.images[name]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    gt_masks = prompt.gt_mask[None,...] if prompt.gt_mask is not None else prompt.gt_masks
    gt_detection = sv.Detections(
        xyxy=sv.detection.utils.mask_to_xyxy(gt_masks),
        mask=gt_masks,
        class_id=np.array([0] * len(gt_masks)),
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