from roboflow import Roboflow
rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("hu178").project("hu164_hca_test_difetti_msa")
dataset = project.version(1).download("coco-segmentation")

train_size = 30
valid_size = 10

points_per_mask = 3