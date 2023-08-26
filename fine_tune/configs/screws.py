from roboflow import Roboflow

rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("segmentation-yolov5").project("yolov5_seg-tm3yy")
dataset = project.version(10).download("coco-segmentation")

# cls_ids = [2] # vertical screws only
tasks = ["point","box"]
num_refinement_steps = 0


vit_patch_embed = True

mask_lora = True
mask_r = 4

train_size = 40 # images
valid_size = 20 # images
points_per_mask = 1

num_refinement_steps = 0