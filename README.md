<div align="center">
  <h1>object-detection-level2-cv-01</h1>
</div>

![pstage](https://user-images.githubusercontent.com/64246382/137623329-2456e10b-276e-40d3-b29c-0a688e12c06e.PNG)
<h6>출처 : aistages </h6>

## :mag: Overview
### Background
> 바야흐로 대량 생산, 대량 소비의 시대, 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 ‘쓰레기대란’,’매립지 부족’ 과 같은 여러 사회 문제를 낳고 있습니다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다. 여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

### Problem definition
> 사진에서 쓰레기를 Detection하여 쓰레기가 사진에 어느 위치에 있는지, 10개의 Class 중 어디에 속하는지를 판별하는 시스템 or 모델

### Development environment
- v100 GPU 서버
- vscode, git, github, wandb, slack

### Evaluation
<img src="https://user-images.githubusercontent.com/64246382/137627632-404ecf72-6244-4128-ae3c-607e8df2a314.PNG" width="600" height="300"/>

## ♻️ Dataset          
<h6>출처 : kr.freepik.com</h6>
<img src="https://user-images.githubusercontent.com/64246382/137628147-122801a1-5492-4ddb-8685-b61428c70f25.jpg" width="700" height="400"/>

### Input
- 전체 이미지 개수 : 9754장 ( Train 4883장, Test 4871장 )
- 10개의 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : ( 1024, 1024 )
- Annotation format : COCO format, YOLO format

### Output
- bbox 좌표, Category, Score 값 리턴
- submission 양식에 맞게 csv 파일 만들어 제출 

### Validation Strategy
- Stratified K-fold

## 🥉 Train Model

### Hybrid Task Cascade
```
● ResNext101_64x4d / FPN / Heavy Augmentation
 - LB score : 0.587
 - Training : SGD, cosine-annealing scheduler, batch size 4
 - Loss: Cross-entropy loss & SmoothL1Loss

● ResNext101_64x4d / FPN / Heavy Augmentation & TTA
 - LB score : 0.601
 - Training: SGD, cosine-annealing scheduler, batch size 4
 - Loss: Cross-entropy loss & SmoothL1Loss
```
### Cascade R-CNN
```
● SwinT / FPN / Soft NMS
 - LB score : 0.561
 - Training: AdamW, cosine-annealing scheduler, batch 16
 - Loss : Cross-entropy loss &SmoothL1Loss

● SwinT / PAFPN / Soft NMS
 - LB score : 0.558
 - Training: AdamW, cosine-annealing scheduler, batch 16
 - Loss : Cross-entropy loss & SmoothL1Loss
```
### EfficientDet
```
● EfficientDet d7x / Flip Augmentation
 - LB score : 0.319
 - Traning: SGD, cosine-annealing scheduler, batch size 2
 - Loss: Cross-entropy loss
```
### YOLO
```
● YOLOv5l
 - LB score : 0.500
 - Training : SGD, mosaic, batch size 16, LambdLR scheduler
 - Loss : BCEWithLogitsLoss

● YOLOv5x
 - LB score : 0.533
 - Training : SGD, mosaic, batch size 16, LambdLR scheduler
 - Loss : BCEWithLogitsLoss

● Yolor
 - LB score : 0.569
 - Training : SGD, mosaic9, batch size 8
 - Loss : Focal Loss
```
