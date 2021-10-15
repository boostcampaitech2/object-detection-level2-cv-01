Yolov5
 - train
   - Python train.py --img 'img size' --batch 'batch size' --epochs 'epoch' --data 'yaml file path' --cfg 'pretrained yaml file path'
 
 - detect
   - Python detect.py --weights 'best.pt file path' --img 'img size' --conf 'confidence t' --source 'test data path' --save-txt --save-conf
 
 - val (Test)
   - Python val.py --data 'y' 



Yolor 
 - train 
   - python train.py --batch-size 8 --img 1024 1024 --data coco.yaml --cfg cfg/yolor_p6.cfg --weight 'yolor_p6.pt' --device 0 --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 200
   
 - detect
   - python detect.py --source /opt/ml/detection/dataset/test/0001.jpg --cfg cfg/yolor_p6.cfg --weights runs/train/yolor_p658/weights/best_ap50.pt --conf 0.5 --img-size 1024 --device 0
     - (source: 검출 할 이미지), (weight: 검출 모델 pt파일), (conf: confidence threshold)
           
 - test
   - python test.py --data data/coco.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.5 --device 0 --cfg cfg/yolor_p6.cfg --weights runs/train/yolor_p666/weights/best_ap50.pt --name yolor_p6_val --subm yes --task test
     - (task: test..), (subm: yes -> submission 파일 만들기)
   
