from pycocotools.coco import COCO
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from tqdm import tqdm
import wandb
from custom_dataset import CustomDataset, CustomValidDataset, collate_fn
from custom_utils import Averager, get_net, load_net

# Albumentation을 이용, augmentation 선언
def get_train_transform():
    return A.Compose([
        # A.RandomSizedBBoxSafeCrop(width=500, height=500, erosion_rate=0.2),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        # A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        # A.RandomSizedBBoxSafeCrop(width=500, height=500, erosion_rate=0.2),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        # A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])
    
# train function
def train_fn(epoch, train_data_loader, optimizer, loss_hist, model, device, clip=35):
    model.train()
    loss_hist.reset()
    mean_loss = 0

    for images, targets, image_ids in tqdm(train_data_loader):

        images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        target = {"bbox": boxes, "cls": labels}

        # calculate loss
        loss, cls_loss, box_loss = model(images, target).values()
        # print(f'<< For Debug >> : {target}')
        # print(f'<< For Debug >> : {loss.shape}, {cls_loss.shape}, {box_loss.shape}')
        loss_value = loss.detach().item()

        # logging
        wandb.log({'train/loss': loss, 'train/cls_loss': cls_loss, 'train/box_loss': box_loss})
        mean_loss += loss_value

        loss_hist.send(loss_value)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

    # result
    mean_loss /= len(train_data_loader)
    print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
    
    # save model
    work_dir = '/opt/ml/detection/work_dir/EffiDet_orig'
    output = os.path.join(work_dir, f'epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), output)
    return mean_loss, epoch
        
            
# valid function
def infer_fn(val_data_loader, model, device):
    outputs = []
    for images, image_ids in tqdm(val_data_loader):   
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
    return outputs


def inference(epoch, val_data_loader, model, device):
    annotation = '/opt/ml/detection/dataset/test.json'
    work_dir = '/opt/ml/detection/work_dir/EffiDet_orig'
    checkpoint_path = os.path.join(work_dir, f'epoch_{epoch+1}.pth')
    model = load_net(checkpoint_path, device)
    score_threshold = 0.1

    model.eval()
    outputs = infer_fn(val_data_loader, model, device)

    prediction_strings = []
    file_names = []
    coco = COCO(annotation)
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_id = coco.getImgIds(imgIds=i)
        image_info = coco.loadImgs(image_id)[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0]*2) + ' ' + str(
                    box[1]*2) + ' ' + str(box[2]*2) + ' ' + str(box[3]*2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    # make submission file
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    output_file = os.path.join(work_dir, f'submission_{epoch}.csv')
    submission.to_csv(output_file, index=None)
    print(submission.head())


def train(num_epochs, train_data_loader, valid_data_loader, optimizer, model, device, clip=35):
    loss_hist = Averager()
    model.train()
    work_dir = '/opt/ml/detection/work_dir/EffiDet_orig'
    os.makedirs(work_dir, exist_ok=True)
    best_loss = 1e9
    
    for epoch in range(num_epochs):
        # train
        mean_loss, epoch = train_fn(epoch, train_data_loader, optimizer, loss_hist, model, device, clip=35)
        # inference
        inference(epoch, valid_data_loader, model, device)
        
        if mean_loss < best_loss:
            print(f'<< Best Epoch : #{epoch} >>')
            best_loss = min(mean_loss, best_loss)
            check_point_name = f'best.pth'
            check_point = os.path.join(work_dir, check_point_name)
            torch.save(model.state_dict(), check_point)
    

def main():

    # init wandb
    wandb.init(project="mmdetection", entity='bagineer')
    wandb.run.name = "efficientdet_d7_SGD_1024_orig"
    wandb.run.save()

    # make datasets
    annotation_train = '/opt/ml/detection/dataset/train.json'
    annotation_valid = '/opt/ml/detection/dataset/test.json'
    data_dir = '/opt/ml/detection/dataset'
    train_dataset = CustomDataset(annotation_train, data_dir, get_train_transform())
    valid_dataset = CustomValidDataset(annotation_valid, data_dir, get_valid_transform())

    # make loaders
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

    model = get_net()
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 30

    train(num_epochs, train_data_loader, valid_data_loader, optimizer, model, device)

if __name__ == '__main__':
    main()