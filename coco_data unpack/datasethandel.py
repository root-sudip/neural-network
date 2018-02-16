from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from config import Config
import utils
import sys
import os
#import model as modellib

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):

        image_dir = os.path.join(dataset_dir, "train2014" if subset == "train"
                                 else "val2014")

        # Create COCO object
        json_path_dict = {
            "train": "annotations/instances_train2014.json",
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict["train"]))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())
            #print(class_ids)

        # # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        #print(image_ids)

        # # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # # Add images
        for i in image_ids:
            path=os.path.join(image_dir, coco.imgs[i]['file_name'])
            #print(path)
            annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=False))
        #print(annotations)
            for j in annotations:
                print(path,',',j['category_id'],',',j['bbox'])
                #print()


dataset_train = CocoDataset()
dataset_train.load_coco(sys.argv[1], "train")
#dataset_train.prepare()