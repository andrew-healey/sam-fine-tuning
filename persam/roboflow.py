import os
import cv2
from pycocotools.coco import COCO
data_dir = "Cancer-Detection-4/train"

new_data_dir = "new_data"
!rm -rf $new_data_dir
!mkdir $new_data_dir

# We designate our *own* "train"/"test" directories--there will be a few images in train/, and most images in test/.
new_train_dir = os.path.join(new_data_dir,"train")
new_test_dir = os.path.join(new_data_dir,"test")
!mkdir $new_train_dir $new_test_dir

ann_file = os.path.join(data_dir,"_annotations.coco.json")
coco=COCO(ann_file)

ann_dir = os.path.join(new_test_dir,"Annotations")
img_dir = os.path.join(new_test_dir,"Images")


for catId in coco.getCatIds():
    cat = coco.cats[catId]
    name = cat["name"]

    cls_ann_dir = os.path.join(ann_dir,name)
    cls_img_dir = os.path.join(img_dir,name)

    for imgId in coco.getImgIds():
        img = coco.imgs[imgId]
        img_cv2 = cv2.imread(os.path.join(data_dir,img["file_name"]))
        annIds = coco.getAnnIds(imgIds=[imgId],catIds=[catId])
        if len(annIds) > 1:
            pass

        img_path = os.path.join(cls_img_dir,f"{imgId}.jpg")
        cv2.imwrite(img_path,img_cv2)
        for annId in annIds:
            ann = coco.anns[annId]
            mask = coco.annToMask(ann)
            mask_path = os.path.join(cls_ann_dir,f"{imgId}.png")
            cv2.imwrite(mask_path,mask*255)