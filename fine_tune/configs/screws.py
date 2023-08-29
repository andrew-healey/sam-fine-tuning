from roboflow import Roboflow

rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("segmentation-yolov5").project("yolov5_seg-tm3yy")
dataset = project.version(10).download("coco-segmentation")

cls_ids = [1,2]
tasks = ["point","box"]


vit_patch_embed = False
mask_lora = False
use_cls_tokens = True

train_size = 40 # images
valid_size = 20 # images
points_per_mask = 1