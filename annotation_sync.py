import os
import shutil
import splitfolders

#splitfolders.fixed("YOLOv7\YOLO_data", output="YOLOv7", seed=42, fixed=(100, 100), oversample=True, group_prefix=None, move=False)

# synchronize train/test/val split with labels for YOLO
data_train_path = "YOLOv7\\train\images"
data_test_path = "YOLOv7\\test\images"
data_val_path = "YOLOv7\\val\images"
anno_train_path = "YOLOv7\\train\labels"
anno_test_path = "YOLOv7\\test\labels"
anno_val_path = "YOLOv7\\val\labels"
anno_path = "YOLOv7\YOLO_annotations"

# if used mask, annotate picture
for a in os.listdir(anno_path):
    anno = a.split("_", 1)
    anno_id = anno[1].split(".")
    # check where label should go after train/test/val split 
    img_id = "fish_%s.png" % anno_id[0]
    if os.path.isfile(os.path.join(data_train_path, img_id)):
        shutil.move(os.path.join(anno_path, a), os.path.join(anno_train_path, a))
    elif os.path.isfile(os.path.join(data_test_path, img_id)):
        shutil.move(os.path.join(anno_path, a), os.path.join(anno_test_path, a))
    elif os.path.isfile(os.path.join(data_val_path, img_id)):
        shutil.move(os.path.join(anno_path, a), os.path.join(anno_val_path, a))
    else:
        print("file was left at annotation folder because no equivalent was found")