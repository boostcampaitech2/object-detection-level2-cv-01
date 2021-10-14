import shutil
from distutils.dir_util import copy_tree

train_save_path = '/opt/ml/detection/yolov5/data/images/train'
val_save_path = '/opt/ml/detection/yolov5/data/images/val'
test_save_path = '/opt/ml/detection/yolov5/data/images/test'
manifest_train_path = "/opt/ml/detection/convert2Yolo/manifest_train.txt"
manifest_val_path = "/opt/ml/detection/convert2Yolo/manifest_val.txt"
test_path = '/opt/ml/detection/dataset/test'

lines = []
with open(manifest_val_path) as f:
    for line in f:
        lines.append(line.rstrip('\n'))

for line in lines:
   shutil.copy(line, val_save_path)

copy_tree(test_path, test_save_path)
