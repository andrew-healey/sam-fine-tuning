
from roboflow import Roboflow
rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("optimaize").project("panneau-dataset")
dataset = project.version(47).download("coco-segmentation")

tasks = ["point","box"]

points_per_mask = 3

train_size = 30 # images
create_valid = True