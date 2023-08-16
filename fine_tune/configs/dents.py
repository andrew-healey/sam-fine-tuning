from roboflow import Roboflow

rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("university-chw4m").project("engenfix-vehicle-background-removedents")
dataset = project.version(5).download("coco-segmentation")