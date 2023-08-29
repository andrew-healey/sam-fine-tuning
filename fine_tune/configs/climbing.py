from roboflow import Roboflow

rf = Roboflow()
project = rf.workspace("roboflow-4rfmv").project("climbing-y56wy")
dataset = project.version(6).download("coco-segmentation")

cls_ids = [1,2,3]

tasks = ["point","box"]

model_size = "vit_h"
vit_patch_embed = False

mask_lora = False

train_size = 40 # images
valid_size = 10 # images
points_per_mask = 3