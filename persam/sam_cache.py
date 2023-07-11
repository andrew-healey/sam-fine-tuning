"""
This file defines logic for caching SAM's encoder outputs.

By caching outputs, we can avoid 90% of the work done during repeated experiments.
This should speed up experiments by a factor of 10x.
"""

import os

from load import *

from segment_anything import SamPredictor

mkdirp("sam_cache")
for sam_type in ["vit_t","vit_h"]:
    mkdirp(os.path.join("sam_cache",sam_type))

def has_cached_output(predictor:SamPredictor,img_path:str):
    """
    Returns True if the SAM encoder output for the given image path exists.
    """
    sam_type = predictor.model.sam_type
    cached_output_path = os.path.join("sam_cache",sam_type,img_path+".pt")
    return os.path.exists(cached_output_path)

import torch
def cache_output(predictor:SamPredictor,img_path:str):
    """
    Caches the SAM encoder output for the given image path.
    """
    sam_type = predictor.model.sam_type
    cached_output_path = os.path.join("sam_cache",sam_type,img_path+".pt")
    cached_output_folder = os.path.dirname(cached_output_path)
    mkdirp(cached_output_folder)

    # Assume the predictor has already been run on the image.
    original_size = predictor.original_size
    input_size = predictor.input_size
    features = predictor.features

    cached_value = (original_size,input_size,features)
    torch.save(cached_value,cached_output_path)

def load_cached_output(predictor:SamPredictor,img_path:str):
    """
    Loads the SAM encoder output for the given image path.
    """
    sam_type = predictor.model.sam_type
    cached_output_path = os.path.join("sam_cache",sam_type,img_path+".pt")

    original_size,input_size,features = torch.load(cached_output_path)

    predictor.reset_image()
    predictor.original_size = original_size
    predictor.input_size = input_size
    predictor.features = features
    predictor.is_image_set = True