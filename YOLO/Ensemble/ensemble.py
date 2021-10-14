#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numpy downgrade가 필요할 수 있음
get_ipython().system('python -m pip install numpy==1.2')


# In[1]:


get_ipython().system('pip install ensemble_boxes')


# In[9]:


import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO


# In[10]:


submission_df = pd.read_csv('./sample/submission_yolov5x.csv')
submission_df = submission_df.drop(['index'], axis=1)
submission_df.to_csv('submission_yolov5x_new.csv', index=False)


# In[40]:


# ensemble csv files
#submission_files = ['./sample_submission.csv'] # submission lists
submission_files = ['./sample/submission_cascade.csv', './sample/submission_cascade_swin.csv',
                    './sample/submission_cascade_swin_soft.csv', './sample/submission_Resnext101_32x4d_DetectoRS.csv',
                    './sample/submission_cascade_swin_panet.csv', './sample/submission_yolor_new.csv', './sample/submission_yolov5x_modified.csv',
                    './sample/wTTA.csv'] # submission lists
submission_df = [pd.read_csv(file) for file in submission_files]


# In[42]:


submission_df


# In[43]:


image_ids = submission_df[0]['image_id'].tolist()


# In[44]:


annotation = '../dataset/test.json'
coco = COCO(annotation)


# In[45]:


prediction_strings = []
file_names = []
iou_thr = 0.6

for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
    
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
    if len(boxes_list):
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)


# In[46]:


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('submission_FINAL_ensemble_wbf.csv', index=False)

submission.head()
submission

