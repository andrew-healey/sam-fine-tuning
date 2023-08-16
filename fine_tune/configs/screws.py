from roboflow import Roboflow

rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("segmentation-yolov5").project("yolov5_seg-tm3yy")
dataset = project.version(10).download("coco-segmentation")

cls_ids = [2] # vertical screws only

valid_patch_embed = True