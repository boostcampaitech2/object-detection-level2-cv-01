#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 라이브러리 및 모듈 import
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import pandas as pd
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
import json
import wandb
from torchsummary import summary


# In[2]:


# CustomDataset class 선언

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        
        # coco annotation 불러오기 (by. coco API)
        self.coco = COCO(annotation)
        self.annotation = annotation
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        with open(self.annotation) as json_file:
            data = json.load(json_file)
        image_id = data['images'][index]['id']
        # image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # boxes (x, y, w, h)
        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # box별 label
        ##################### label에 1을 더해줘서 0~9를 1~10으로 ####################
        labels = np.array([x['category_id'] + 1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([image_id]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break
            
        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())


# In[3]:


# valid mAP 계산을 위한 gt 생성
GT_JSON = '/opt/ml/detection/dataset/val_kfold1.json'
coco = COCO(GT_JSON)
gt = []
current_id = 0
for image_id in coco.getImgIds():
        
    image_info = coco.loadImgs(image_id)[0]
    annotation_id = coco.getAnnIds(imgIds=image_info['id'])
    annotation_info_list = coco.loadAnns(annotation_id)
        
    file_name = image_info['file_name']
    ##################### label에 1을 더해줘서 0~9를 1~10으로 ####################
    for annotation in annotation_info_list:
        gt.append([str(current_id), str(annotation['category_id']+1),
                float(annotation['bbox'][0]),
                float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                float(annotation['bbox'][1]),
                float(annotation['bbox'][1]) + float(annotation['bbox'][3])])
                # 1024
    current_id += 1


# In[4]:


# Albumentation을 이용, augmentation 선언
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        # A.RandomCrop(200, 200, always_apply=False, p=0.5),
        # A.ToGray(),
        # A.MotionBlur(),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
        # A.MedianBlur(blur_limit=5, always_apply=False, p=0.5),
        # A.Blur(blur_limit=7, always_apply=False, p=0.5),
        # A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        # A.Resize(1024, 1024),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'min_visibility': 0.1, 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        # A.Resize(1024, 1024),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# In[5]:


# loss 추적
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))


# In[6]:


# Effdet configcheckpoint_path
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py

# Effdet config를 통해 모델 불러오기
def get_net(checkpoint_path=None):
    
    config = get_efficientdet_config('tf_efficientdet_d7x') # 여기서 모델 선택
    config.num_classes = 10
    config.image_size = (1024,1024)
    
    config.soft_nms = False
    config.max_det_per_image = 50
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        for key in checkpoint.copy().keys():
            if 'model.' in key:
                checkpoint[key[6:]] = checkpoint[key]
                del checkpoint[key]
            if 'anchor' in key:
                del checkpoint[key] 
        net.load_state_dict(checkpoint)
        
    return DetBenchTrain(net)
    
def load_net(checkpoint_path, device):
    config = get_efficientdet_config('tf_efficientdet_d7x')
    # get_net에서의 모델과 같아야 함
    config.num_classes = 10
    config.image_size = (1024,1024)
    
    config.soft_nms = False
    config.max_det_per_image = 50
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)
    
# train function
def train_fn(num_epochs, train_data_loader, valid_data_loader, optimizer, scheduler, model, device, clip=35):
    loss_hist = Averager()
    wandb.init(project="EfficientDet", name="1024_SGD_start_with_43epoch",reinit=True)
    wandb.watch(model, log_freq=100)
    model.train()

    for epoch in range(num_epochs):
        loss_hist.reset()
        model.train()
        
        for images, targets, image_ids in tqdm(train_data_loader):
            
                images = torch.stack(images) # bs, ch, w, h - 16, 3, 1024, 1024
                images = images.to(device).float()
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels}
                # calculate loss
                loss, cls_loss, box_loss = model(images, target).values()
                loss_value = loss.detach().item()
                
                loss_hist.send(loss_value)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                # grad clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()
                scheduler.step()
        

        print(f"Epoch #{epoch} loss: {loss_hist.value}")
        torch.save(model.state_dict(), f'epoch_{epoch}_1024.pth')

        # validation 단계
        model.eval()
        with torch.no_grad():
            predict_model = load_net(f'epoch_{epoch}_1024.pth', device)
            outputs = valid_fn(valid_data_loader, predict_model, device)

            score_threshold = 0.05
            prediction_list = []
            tmp_image_name = 0

            # output을 기반으로 mean_average_precision_for_boxes 에 들어갈 수 있는 형식으로 변경
            for i, output in enumerate(outputs):
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    if score > score_threshold:
                        prediction_list.append([str(tmp_image_name), str(int(label)), str(score), str(box[0]), str(box[2]), str(box[1]), str(box[3])])
                tmp_image_name += 1
            mean_ap, average_precisions = mean_average_precision_for_boxes(gt, prediction_list, iou_threshold=0.5)
            
            # print(prediction_list)
            # print(mean_ap)

        wandb.log({
            "loss": loss_hist.value,
            "mAP50": mean_ap
        })

# valid function
def valid_fn(val_data_loader, model, device):
    outputs = []
    for images, _, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)       
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]
                            })
    return outputs


# In[7]:


def main():
    train_annotation = '/opt/ml/detection/dataset/train.json'
    valid_annotation = '/opt/ml/detection/dataset/val_kfold1.json'
    data_dir = '../dataset'
    train_dataset = CustomDataset(train_annotation, data_dir, get_train_transform())
    valid_dataset = CustomDataset(valid_annotation, data_dir, get_valid_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = get_net('/opt/ml/detection/mmdetection_new/epoch_43_1024_0.699.pth') # 체크포인트가 있다면 경로를 입력
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50

    loss = train_fn(num_epochs, train_data_loader, valid_data_loader, optimizer, scheduler, model, device)


# In[ ]:


if __name__ == '__main__':
    main()

