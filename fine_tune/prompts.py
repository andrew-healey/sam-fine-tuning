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
        raise DeprecationWarning("ClassInfo is deprecated. Use gt_cls and gt_clss instead.")
        if len(self.gt_class) > 0:
            assert np.max(self.gt_class) < self.num_classes,f"gt_class must be less than num_classes. But gt_class = {self.gt_class} and num_classes = {self.num_classes}"

@dataclass
class Prompt:
    points: ndarray = None
    labels: ndarray = None
    box: Box = None
    mask: Union[BoolMask,FloatMask] = None # TODO: also allow FloatMask

    gt_mask: BoolMask = None
    gt_masks: BoolMask = None

    gt_cls: np.ndarray = None
    gt_clss: np.ndarray = None

    context: List[ContextPair] = None

    multimask: bool = False

    mask_loss: bool = True

    testing: bool = False

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
            assert self.mask.dtype == bool or self.mask.dtype == np.float32,f"mask must have dtype bool or float, not {self.mask.dtype}"

        assert (self.gt_mask is None) != (self.gt_masks is None),"Prompt must have exactly one of gt_mask or gt_masks"
    
        if self.gt_cls is not None:
            assert self.gt_mask is not None,"Prompt must have gt_mask if gt_cls is not None"
        if self.gt_clss is not None:
            assert self.gt_masks is not None,"Prompt must have gt_masks if gt_clss is not None"
