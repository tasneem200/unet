#!/usr/bin/env python
# coding: utf-8

# In[41]:


import cloudvolume
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import h5py
import shutil
import random
from imageryclient import ImageryClient


# In[40]:



#Simple Example of one file
#Initialize cloudvolume connection
img_src = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'
img_cv = cloudvolume.CloudVolume(img_src, use_https=True)

img_cv.bounds

img_cv.scale

#Look at the data bounds

img_cv.info

ctr = [229233/2,86699/2,26485]
image_size = 200
img = img_cv[ctr[0]-image_size:ctr[0]+image_size,ctr[1]-image_size:ctr[1]+image_size,ctr[2]-1:ctr[2]]

plt.imshow(np.squeeze(img), cmap=plt.cm.gray)


# In[27]:


#Example with volume files
tokenfile = '../chunkedgraph-secret.json'
with open(tokenfile) as f:
      tokencfg = json.load(f)
token = tokencfg['token']


img_src = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'
img_cv = cloudvolume.CloudVolume(img_src, use_https=True)
seg_src = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie3_v1'
seg_cv = cloudvolume.CloudVolume(seg_src, use_https=True,secrets = token)

A = pd.read_csv('annotations.csv')
for i,row in A.iterrows():
    mylist = row['Coordinate 1'].strip('(').strip(')').split(',')
    mylist = list(map(int, mylist))
    edlist = row['Ellipsoid Dimensions'].split(' × ')
    edlist = list(map(int, edlist))
    print(mylist, edlist)
    ctr = np.array(mylist)/[2,2,1] # to deal with scale issues 
    image_size = np.array(edlist)/[4,4,2] # 2 to make it half for radius and 4 to deal with scale on x,y as above
    img = img_cv[ctr[0]-image_size[0]:ctr[0]+image_size[0],ctr[1]-image_size[1]:ctr[1]+image_size[1],ctr[2]-image_size[2]:ctr[2]+image_size[2]]
    seg = seg_cv[ctr[0]-image_size[0]:ctr[0]+image_size[0],ctr[1]-image_size[1]:ctr[1]+image_size[1],ctr[2]-image_size[2]:ctr[2]+image_size[2]]
    break


# In[28]:


index_lis = [i for i in range(1, 118)]
np.random.shuffle(index_lis)
print(index_lis)
train_index = index_lis[:93]
print(train_index)
test_index = index_lis[93:118]
print(test_index)
training_set = A.iloc[train_index]
testing_set = A.iloc[test_index]
print(training_set)
print(testing_set)


# In[38]:


#Using Imagery Client
from PIL import Image 
tokenfile = '../chunkedgraph-secret.json'
with open(tokenfile) as f:
      tokencfg = json.load(f)
token = tokencfg['token']


img_src = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em'
seg_src = 'graphene://https://minnie.microns-daf.com/segmentation/table/minnie3_v1'
ic=ImageryClient(image_source = img_src,
                 segmentation_source=seg_src,auth_token=token)

# Training images 
A = pd.read_csv('annotations.csv')
A_train = training_set
index2 = 0 
for index,row in A_train.iterrows():
    print(index)
    mylist = row['Coordinate 1'].strip('(').strip(')').split(',')
    mylist = list(map(int, mylist))
    edlist = row['Ellipsoid Dimensions'].split(' × ')
    edlist = list(map(int, edlist))
#     print(mylist, edlist)

    ctr = np.array(mylist)/[2,2,1] # to deal with scale issues 
    image_size = (np.array(edlist)/[4,4,2])# 2 to make it half for radius and 4 to deal with scale on x,y as above
    image_size = image_size * [8,8,1]
    bounds=[ctr-image_size,
        ctr+image_size]

    imgvol, segdict = ic.image_and_segmentation_cutout(bounds,split_segmentations=True,root_ids=[864691136534887842], image_mip = 3, segmentation_mip = 3) 

    # Resizing images and masks to 128x128
    resized_images = []
    resized_seg = []
    
    for n in range(0, 100):
        index2 = index2 + 1
        startx = random.randint(0, (imgvol.shape[0]-128))
        starty = random.randint(0, (imgvol.shape[1]-128))
        startz = random.randint(0, (imgvol.shape[2]-1))
        new = imgvol[startx:startx+128, starty:starty+128, startz:startz+1] 
        newseg = segdict[864691136534887842][startx:startx+128, starty:starty+128, startz:startz+1]
        resized_images.append(new)
        resized_seg.append(newseg)
        filename = "newdata/file_%06d.png"%index2
        segfilename = "newdata/seg_%06d.png"%index2
        
        with open(filename, 'wb') as f:
            im = Image.fromarray(np.squeeze(new).astype(np.uint8))
            im.save('training2/file%06d.png'%index2)
            im.show()
        with open(segfilename, 'wb') as s:
            i = Image.fromarray((np.squeeze(newseg)*255).astype(np.uint8))
            i.save('training2/mask%06d.png'%index2)
            i.show()
        
print("Done!")    


# In[37]:


# Testing images 
index2 = 0
A_test = testing_set
for index,row in A_test.iterrows():
    print(index)
    mylist = row['Coordinate 1'].strip('(').strip(')').split(',')
    mylist = list(map(int, mylist))
    edlist = row['Ellipsoid Dimensions'].split(' × ')
    edlist = list(map(int, edlist))

    ctr = np.array(mylist)/[2,2,1] # to deal with scale issues 
    image_size = (np.array(edlist)/[4,4,2]) # 2 to make it half for radius and 4 to deal with scale on x,y as above
    image_size = image_size * [8,8,1]
    bounds=[ctr-image_size,
        ctr+image_size]

    imgvol, segdict = ic.image_and_segmentation_cutout(bounds,split_segmentations=True,root_ids=[864691136534887842],image_mip = 3,segmentation_mip = 3) 

    # Resizing images and masks to 128x128
    resized_images = []
    resized_seg = []
    for n in range(0, 100):
        index2 = index2 + 1
        startx = random.randint(0, (imgvol.shape[0]-128))
        starty = random.randint(0, (imgvol.shape[1]-128))
        startz = random.randint(0, (imgvol.shape[2]-1))
        new = imgvol[startx:startx+128, starty:starty+128, startz:startz+1] 
        newseg = segdict[864691136534887842][startx:startx+128, starty:starty+128, startz:startz+1]
        resized_images.append(new)
        resized_seg.append(newseg)
        filename = "newdata/file_%06d.png"%index2
        segfilename = "newdata/seg_%06d.png"%index2
        
        with open(filename, 'wb') as f:
            im = Image.fromarray(np.squeeze(new).astype(np.uint8))
            im.save('testing2/file%06d.png'%index2)
            im.show()
        with open(segfilename, 'wb') as s:
            i = Image.fromarray((np.squeeze(newseg)*255).astype(np.uint8))
            i.save('testing2/mask%06d.png'%index2)
            i.show()    
print("Done!")

