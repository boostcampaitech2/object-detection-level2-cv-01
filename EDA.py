#!/usr/bin/env python
# coding: utf-8

# In[856]:


import pandas as pd
import numpy as np
import json
from collections import Counter

from pandas.io.json import json_normalize
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import rgb2hex

with open('/opt/ml/detection/dataset/train.json') as json_data:
    data = json.load(json_data)

df_imgs = pd.DataFrame(data['images'])
df_cats = pd.DataFrame(data['categories'])
df_annos = pd.DataFrame(data['annotations'])
category = ['General Trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']
label_colors = ['black', 'darkcyan', 'sienna', 'gray', 'navy', 'chartreuse', 'firebrick', 'seagreen', 'darkorchid', 'olivedrab']


# In[503]:


def set_text(ax, data, size=10):
    rects = ax.patches
    labels = data
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", size=size
        )


# ### Area Distribution

# In[903]:


img_areas = df_annos['area'].astype(int).value_counts()
img_areas = dict(img_areas)
img_areas = sorted(img_areas.items(), key=lambda x: x[0])

areas, cnts = list(zip(*img_areas))

print(np.min(areas))
print(np.max(areas))
print(np.mean(areas))
print(np.median(areas))

fig, ax = plt.subplots(figsize=(12, 7))
# ax.plot(areas, cnts)
ax.plot(areas, cnts)
plt.xscale('log')
plt.show()


# ### Area Range Distribution

# In[904]:


area_rng = df_annos['area'].astype(int)
small = area_rng < 32*32
medium = (area_rng >= 32*32) & (area_rng < 96*96)
large = (area_rng >= 96*96) & (area_rng < 1e10)

print(len(area_rng))
print(len(area_rng[small]) + len(area_rng[medium]) + len(area_rng[large]))

x = ['small', 'medium', 'large']
y = [len(area_rng[small]), len(area_rng[medium]), len(area_rng[large])]

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x, y)
ax.text(-0.4, max(y)*0.9, f'all boxes = {sum(y)}', size=20)
set_text(ax, y, size=15)
plt.show()


# ### box ratio

# In[820]:


def get_box_ratio(bbox):
    x_min, y_min, w, h = bbox
    ratio = w / h
    return ratio


# In[902]:


bbox_list = df_annos['bbox'].tolist()
ratio_list = [get_box_ratio(bbox) for bbox in bbox_list]

counter = sorted(Counter(ratio_list).items(), key=lambda x: x[0])
ratios, cnts = list(zip(*counter))

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Box Ratio Distribution')
ax.plot(ratios, cnts)
plt.show()


# ### x_c, y_c

# In[894]:


def get_center_point(bbox):
    x_min, y_min, w, h = bbox
    return (x_min + w/2, y_min + h/2)


# In[901]:


bbox_list = df_annos['bbox'].tolist()
cp_list = [get_center_point(bbox) for bbox in bbox_list]
xcs, ycs = list(zip(*cp_list))

fig, ax = plt.subplots(figsize=(12,12))
fig.suptitle('BBox Center Points')
ax.scatter(xcs, ycs, s=1, color='red')
plt.show()


# ### boxex per category

# In[908]:


for cat in range(len(category)):
    df_cat = df_annos[df_annos['category_id'] == cat]

    area_rng = df_cat['area'].astype(int)
    small = area_rng < 32*32
    medium = (area_rng >= 32*32) & (area_rng < 96*96)
    large = (area_rng >= 96*96) & (area_rng < 1e10)
    
    x = ['small', 'medium', 'large']
    y = [len(area_rng[small]), len(area_rng[medium]), len(area_rng[large])]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    plt.suptitle(category[cat], size=20)
    ax[0].bar(x, y, color=label_colors[cat])
    set_text(ax[0], y, size=13)
    
    base_color = label_colors[cat]
    scaler = [0.1, 0.2, 0.3]
    new_color = colors.to_rgb(base_color)
    new_color = [rgb2hex(tuple(min(max(val+s, 0), 1) for val in new_color)) for s in scaler]
    
    # for i,j in zip(x,y):
    #     ax.annotate(str(j), xy=(i,j), size=15)
    ax[0].text(-0.4, max(y)*0.9, f'all boxes = {sum(y)}', size=20)
    ax[0].set_aspect(3/max(y))
    # plt.tight_layout()
    plt.ylim(top=max(y)*1.1)
    patches, texts, pcts = ax[1].pie(y, labels=['small', 'medium', 'large'],
               wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
              autopct='%1.1f%%', colors=new_color)
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='firebrick', weight="bold")
    plt.show()


# ### categories

# In[711]:


df_cat = df_annos.value_counts('category_id')

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(df_cat.keys(), df_cat.values, color=label_colors)
plt.xticks(np.arange(10), category)
set_text(ax, df_cat.values, size=13)
plt.show()


# ### box plot

# In[443]:


data = [df_annos[df_annos['category_id'] == cid]['area'] for cid in range(len(category))]

fig, ax = plt.subplots(figsize=(12, 7))
bp = ax.boxplot(data, meanline=True, showmeans=True, widths=0.3)
# ax.hlines(y=1000, xmin=1, xmax=10, linewidth=2, color='r')
ax.axhline(y=32*32, xmin=0, xmax=1, color='r', linestyle=':')
ax.axhline(y=96*96, xmin=0, xmax=1, color='b', linestyle=':')
plt.yscale('log')
plt.xlabel('category')
plt.ylabel('area')
plt.xticks(np.arange(1, 11), category)
plt.setp(bp['fliers'], markersize=1.0)
plt.setp(bp['medians'], color='r', linewidth=2, drawstyle='steps-post')

# medians = [bp['medians'][i].get_ydata()[0] for i in range(len(category))]
# # print(bp['medians'][0].get_ydata()[0])
# print(medians)

for line in bp['medians']:
    x, y = line.get_xdata()
    med = int(line.get_ydata()[0])
    ax.annotate(str(med), xy=(x-0.1, med), size=11)
plt.show()


# def add_n_obs(df,group_col,y):
#     medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}
#     xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
#     n_obs = df.groupby(group_col)[y].size().values
#     for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
#         plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')


# ## Inference result

# In[813]:


def get_area(box):
    box = list(map(float, box[:, 2:].squeeze()))
    width = box[2] - box[0]
    height = box[3] - box[1]
    return int(width*height)

color_list = ['royalblue', 'gold', 'crimson', 'aquamarine', 'tab:red', 'tab:blue', 'tab:cyan', 'tab:purple', 'tab:gray', 'limegreen']


# ### Cascade_Resnext101_32x4d_DetectoRS_537

# In[814]:


df = pd.read_csv('./data/Cascade_Resnext101_32x4d_DetectoRS_537.csv')

total_areas = []
for i in range(len(df)):
    boxes = df.loc[i, 'PredictionString']

    # PredictionString = (label, score, xmin, ymin, xmax, ymax), .....
    box_list = boxes.split()
    box_list = np.reshape(box_list, (-1, 1, 6))

    box_areas = [get_area(box) for box in box_list]
    total_areas.extend(box_areas)
    # print(box_areas)
    
total_areas = np.asarray(total_areas)
small = np.where(total_areas < 32*32)
medium = np.where((total_areas >= 32*32) & (total_areas < 96*96))
large = np.where((total_areas >= 96*96) & (total_areas < 1e10))
print(len(total_areas))
area_range = [len(total_areas[small]), len(total_areas[medium]), len(total_areas[large])]

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
plt.suptitle('Cascade_Resnext101_32x4d_DetectoRS_537', size=20)
ax[0].bar(['small', 'medium', 'large'], area_range, color=color_list[:3])
ax[1].pie(area_range, labels=['small', 'medium', 'large'],
           wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
          autopct='%1.0f%%', colors=color_list[:3])
set_text(ax[0], area_range, size=13)
ax[0].set_aspect(3/max(area_range))
ax[0].text(-0.4, max(area_range)*0.95, f'all boxes = {sum(area_range)}', size=20)
plt.show()


# ### cascadeRCNN_SGD_resnet50_390

# In[815]:


df = pd.read_csv('./data/cascadeRCNN_SGD_resnet50_390.csv')

total_areas = []
for i in range(len(df)):
    boxes = df.loc[i, 'PredictionString']
    
    try:
        box_list = boxes.split()
    except:
        continue

    # PredictionString = (label, score, xmin, ymin, xmax, ymax), .....
    # box_list = boxes.split()
    box_list = np.reshape(box_list, (-1, 1, 6))

    box_areas = [get_area(box) for box in box_list]
    total_areas.extend(box_areas)
    # print(box_areas)
    
total_areas = np.asarray(total_areas)
small = np.where(total_areas < 32*32)
medium = np.where((total_areas >= 32*32) & (total_areas < 96*96))
large = np.where((total_areas >= 96*96) & (total_areas < 1e10))
print(len(total_areas))
area_range = [len(total_areas[small]), len(total_areas[medium]), len(total_areas[large])]

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
plt.suptitle('cascadeRCNN_SGD_resnet50_390', size=20)
ax[0].bar(['small', 'medium', 'large'], area_range, color=color_list[:3])
ax[1].pie(area_range, labels=['small', 'medium', 'large'],
           wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
          autopct='%1.0f%%', colors=color_list[:3])
set_text(ax[0], area_range, size=13)
ax[0].set_aspect(3/max(area_range))
ax[0].text(-0.4, max(area_range)*0.95, f'all boxes = {sum(area_range)}', size=20)
plt.show()


# ### cascadeRCNN_Swin_FPN_500

# In[816]:


df = pd.read_csv('./data/cascadeRCNN_Swin_FPN_500.csv')

total_areas = []
for i in range(len(df)):
    boxes = df.loc[i, 'PredictionString']
    
    try:
        box_list = boxes.split()
    except:
        continue

    # PredictionString = (label, score, xmin, ymin, xmax, ymax), .....
    # box_list = boxes.split()
    box_list = np.reshape(box_list, (-1, 1, 6))

    box_areas = [get_area(box) for box in box_list]
    total_areas.extend(box_areas)
    # print(box_areas)
    
total_areas = np.asarray(total_areas)
small = np.where(total_areas < 32*32)
medium = np.where((total_areas >= 32*32) & (total_areas < 96*96))
large = np.where((total_areas >= 96*96) & (total_areas < 1e10))
print(len(total_areas))
area_range = [len(total_areas[small]), len(total_areas[medium]), len(total_areas[large])]

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
plt.suptitle('cascadeRCNN_Swin_FPN_500', size=20)
ax[0].bar(['small', 'medium', 'large'], area_range, color=color_list[:3])
ax[1].pie(area_range, labels=['small', 'medium', 'large'],
           wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
          autopct='%1.0f%%', colors=color_list[:3])
set_text(ax[0], area_range, size=13)
ax[0].set_aspect(3/max(area_range))
ax[0].text(-0.4, max(area_range)*0.95, f'all boxes = {sum(area_range)}', size=20)
plt.show()


# ### YoloR

# In[817]:


df = pd.read_csv('./data/yolor_submission7.csv')

total_areas = []
for i in range(len(df)):
    boxes = df.loc[i, 'PredictionString']
    
    try:
        box_list = boxes.split()
    except:
        continue

    # PredictionString = (label, score, xmin, ymin, xmax, ymax), .....
    # box_list = boxes.split()
    box_list = np.reshape(box_list, (-1, 1, 6))

    box_areas = [get_area(box) for box in box_list]
    total_areas.extend(box_areas)
    # print(box_areas)
    
total_areas = np.asarray(total_areas)
small = np.where(total_areas < 32*32)
medium = np.where((total_areas >= 32*32) & (total_areas < 96*96))
large = np.where((total_areas >= 96*96) & (total_areas < 1e10))
print(len(total_areas))
area_range = [len(total_areas[small]), len(total_areas[medium]), len(total_areas[large])]

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
plt.suptitle('yolor_submission7', size=20)
ax[0].bar(['small', 'medium', 'large'], area_range, color=color_list[:3])
ax[1].pie(area_range, labels=['small', 'medium', 'large'],
           wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
          autopct='%1.0f%%', colors=color_list[:3])
set_text(ax[0], area_range, size=13)
ax[0].set_aspect(3/max(area_range))
ax[0].text(-0.4, max(area_range)*0.95, f'all boxes = {sum(area_range)}', size=20)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # test

# In[3]:


df_imgs


# In[4]:


df_cats


# In[5]:


df_annos


# In[39]:


image_dir = '/opt/ml/detection/dataset/'
img_id_iter = iter(df_imgs['id'])


# In[40]:


img_id = next(img_id_iter)
img_name = df_imgs['file_name'][img_id]

print(img_name)


# In[41]:


from PIL import Image
import os
print(os.listdir(image_dir))

img_file = os.path.join(image_dir, img_name)
Image.open(img_file)


# In[76]:


indices = df_annos['image_id'] == 0
print(df_annos[indices])
print(df_annos.loc[indices, 'bbox'].values)

x_min, y_min, width, height = df_annos.loc[indices, 'bbox'].values[0]
print(x_min, y_min, width, height)
print(width * height)


# In[764]:


import matplotlib
bcode = matplotlib.colors.cnames["blue"]
print(bcode)

from matplotlib import colors

print(colors.to_rgb('blue'))
a = colors.to_rgb('tab:red')
print(a)
a = (a[0]*0.95, a[1]*0.95, a[2]*0.95)
a = tuple(val*0.9 for val in a)
a

from matplotlib.colors import rgb2hex
b = rgb2hex(a)
print(b)


# In[ ]:




