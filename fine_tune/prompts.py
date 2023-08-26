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
class ClassInfo:
    num_classes: int
    gt_class: np.ndarray # must have same length as gt_masks

    def __post_init__(self):
        if len(self.gt_class) > 0:
            assert np.max(self.gt_class) < self.num_classes,f"gt_class must be less than num_classes. But gt_class = {self.gt_class} and num_classes = {self.num_classes}"

@dataclass
class Prompt:
    points: ndarray = None
    labels: ndarray = None
    box: Box = None
    mask: BoolMask = None # TODO: also allow FloatMask

    gt_mask: BoolMask = None
    gt_masks: BoolMask = None

    context: List[ContextPair] = None
    cls_info: ClassInfo = None

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

        if self.cls_info is not None:
            assert self.cls_info.num_classes > 0,f"num_classes must be positive, not {self.cls_info.num_classes}"

            if self.gt_mask is not None:
                assert self.cls_info.gt_class.shape == (1,),f"gt_class must have shape (1,), not {self.cls_info.gt_class.shape}"
            else:
                assert len(self.cls_info.gt_class) == len(self.gt_masks),f"gt_classes must have the same length as gt_masks, not {len(self.cls_info.gt_class)} and {len(self.gt_masks)}"


from torch import Tensor
def make_refinement_prompt(pred_mask: Tensor, gt_binary_mask: Tensor):
    return Prompt(
        mask=np.array(pred_mask.cpu().detach() > 0),
        gt_mask=np.array(gt_binary_mask.cpu().detach() > 0),
        multimask=False
    )