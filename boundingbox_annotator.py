import os
import torch
from torchvision.io import read_image
import cv2
import numpy as np
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes

def convert_BB_coordinates_YOLO(bbox):
    # 0 = x1, 1 = y1 (lower left corner), 2 = x2, 3 = y2 (upper left)
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    mid_x = width / 2 + bbox[0]
    mid_y = height / 2 + bbox[1]
    annotation = [cl, mid_x, mid_y, width, height]
    return annotation

# convert masks to bounding boxes for YOLOv7 annotations 
# YOLO format is <object-class-id> <x> <y> <width> <height>
# iterate over all masks in one class directory
mask_path = "annotations\mask_01"
img_path = "data\\fish_01"
anno_path = "annotations\labels_01"
cl = "01"

# if used mask, annotate picture
for m in os.listdir(mask_path):
    mask_id = m.split("_", 1)
    # check if mask should be used for fish
    iname = "fish_%s" % mask_id[1]
    if os.path.isfile(os.path.join(img_path, iname)):
        mask = read_image(os.path.join(mask_path, m))
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        '''convert to bbox coordinates
        The boxes are in x1, y1, -> lower left and x2, y2 -> upper right format with 0 <= x1 < x2 and 0 <= y1 < y2.'''
        boxes = masks_to_boxes(masks)
        yolo_annot = convert_BB_coordinates_YOLO(boxes.flatten().tolist())
        id = mask_id[1].split(".")
        f = open(os.path.join(anno_path, "label_%s.txt" % id[0]), "w")
        line = ' '.join(map(str, yolo_annot))
        f.write(line)
        f.close()
        print("wrote new label for %s into label directory % mask_id")
    else:
        os.remove(os.path.join(mask_path, m))
        print("removed mask %s from directory % mask_id")




