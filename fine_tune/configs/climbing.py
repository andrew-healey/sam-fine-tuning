from roboflow import Roboflow

rf = Roboflow()
project = rf.workspace("roboflow-4rfmv").project("climbing-y56wy")
dataset = project.version(6).download("coco-segmentation")

cls_ids = [1,2,3]

tasks = ["point","box"]

model_size = "vit_t"
vit_patch_embed = False

mask_lora = False
mask_r = 1

train_size = None # images
valid_prompts = 200 # prompts
points_per_mask = [1,10,10]

lr = 8e-4

create_valid = False