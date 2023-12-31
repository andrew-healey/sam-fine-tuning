from typing import Tuple, Optional, Dict, Generator,Union

import sys
import os
# add to front of syspath
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "segment-anything"))

from segment_anything import sam_model_registry, SamPredictor

import torch

import cv2

def mkdirp(dir: str):
    os.makedirs(dir, exist_ok=True)


import shutil

def rmrf(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def cp(src: str, dst: str):
    shutil.copy(src, dst)

from .sam_cache import *

def load_image(
    predictor: SamPredictor,
    image_path: str,
    mask_path: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_cache:
        if has_cached_output(predictor, image_path):
            load_cached_output(predictor, image_path)
        else:
            predictor.set_image(image)
            cache_output(predictor, image_path)
    else:
        predictor.set_image(image)

    if mask_path is None:
        mask = None
    else:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = predictor.preprocess_mask(mask)

    return image, mask


def load_predictor(sam_type: str = "vit_h", *args, **kwargs) -> SamPredictor:
    if sam_type == "vit_h":
        sam_type, sam_ckpt = "vit_h", "weights/sam_vit_h_4b8939.pth"
        sam = sam_model_registry[sam_type](*args,**kwargs,checkpoint=sam_ckpt).cuda()
    elif sam_type == "vit_t":
        sam_type, sam_ckpt = "vit_t", "weights/mobile_sam.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(kwargs)
        sam = sam_model_registry[sam_type](*args,**kwargs,checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    sam.sam_type = sam_type

    predictor = SamPredictor(sam)

    return predictor


import glob


def load_images_in_dir(dir: str) -> Generator[Tuple[str, str], None, None]:
    image_paths = glob.glob(os.path.join(dir, "*.*"))
    for image_path in image_paths:
        image_name = os.path.basename(image_path).split(".")[0]
        yield image_name, image_path


import numpy as np

def save_mask(mask: Union[np.ndarray,torch.Tensor], mask_path: str):
    if isinstance(mask, torch.Tensor):
        mask = mask.to(torch.float32).cpu().numpy()
    mask_colors = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    mask_colors[:,:,0] = 128
    mask_colors *= mask[:,:,None]
    mask_colors = cv2.cvtColor(mask_colors, cv2.COLOR_BGR2RGB)
    cv2.imwrite(mask_path, mask_colors)

def zip_globs(*globs):
    evalled = [glob.glob(g) for g in globs]
    zipped = zip(*evalled)
    return zipped


def load_dirs(
    ref_img_glob: str, ref_mask_glob: str, test_img_dir_glob: str, output_super_dir: str
):
    for ref_img_path, ref_mask_path, test_img_dir in zip_globs(
        ref_img_glob, ref_mask_glob, test_img_dir_glob
    ):
        output_dir = os.path.join(output_super_dir, os.path.basename(test_img_dir))
        yield ref_img_path, ref_mask_path, test_img_dir, output_dir
