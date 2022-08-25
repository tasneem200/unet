#!/usr/bin/env python
# coding: utf-8

# In[127]:


import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


# In[125]:


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
with tf.device("cpu:0"):
    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 1

    TRAIN_PATH = 'training2'
    TEST_PATH = 'testing2'

    num_train_ids = 5200
    num_test_ids = 2400
    

    X_train = np.zeros((num_train_ids, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((num_train_ids, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# In[ ]:


print('Resizing training images and masks')
for image in range(1, 5200):   
    path = TRAIN_PATH 
    img = imread(path + '/file%06d.png'%image) 
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                    preserve_range=True), axis=-1)
    
    X_train[image] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = imread(path + '/mask%06d.png'%image)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                    preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)  
                
    Y_train[image] = mask   
print("Done!")


# In[109]:


img.shape


# In[110]:


# test images
X_test = np.zeros((num_test_ids, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for image in range(1, 2400):   
    path = TEST_PATH 
    img = imread(path + '/file%06d.png'%image)
    sizes_test.append([img.shape[0], img.shape[1]])
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                    preserve_range=True), axis=-1)
    X_test[image] = img


print('Done!')


# In[111]:


image_x = random.randint(1, 5200)
print(image_x)
imshow(X_train[image_x], cmap = 'gray')
plt.show()
imshow((np.squeeze(Y_train[image_x])).astype(np.uint8))
plt.show()
plt.clim(0, 255) 
m = (X_train[image_x]) 
print(np.min(m))
print(np.max(m))
print(np.std(m))


# In[91]:


m = (X_train[120]) 
print(np.min(m))
print(np.max(m))
print(np.std(m))

n = (X_train[100]) 
print(np.min(n))
print(np.max(n))
print(np.std(n))


# In[92]:


imshow((np.squeeze(Y_train[image_x])).astype(np.uint8))


# In[112]:


#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) 
#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


# In[113]:


outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=2E-4)
model.summary()


# In[114]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[123]:


#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_imageseg1.h5', verbose=1, save_best_only=True)


callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'), 
             tf.keras.callbacks.TensorBoard(log_dir='logs'), checkpointer]

results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=502, epochs=10, callbacks=callbacks)


# In[116]:


print(callbacks[0])
idx = random.randint(0, 2100)
 
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.7)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.7):], verbose=1)
preds_test = model.predict(X_test, verbose=1)
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8) 


# In[ ]:


preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8) 


# In[98]:


# Perform a sanity check on some random training samples
ix = random.randint(1, 3639)
print(ix)
imshow(X_train[ix], cmap = 'gray')
plt.show()
imshow((np.squeeze(Y_train[ix])).astype(np.uint8))
plt.show()
imshow((np.squeeze(preds_train_t[ix])).astype(np.uint8))
plt.show()


# In[105]:


# Perform a sanity check on some random validation samples
ix = random.randint(1, 300)
print(ix)
imshow(X_train[int(X_train.shape[0]*0.7):][ix], cmap='gray')
plt.show()
imshow((np.squeeze(Y_train[int(Y_train.shape[0]*0.7):][ix])).astype(np.uint8))
plt.show()
imshow((np.squeeze(preds_val_t[ix])).astype(np.uint8))
plt.show()


# In[103]:


# Plot loss values for each epoch in the training data 
plt.plot(results.epoch, results.history['loss'], "-b", label = "training loss")
plt.plot(results.epoch, results.history['val_loss'], "-r", label = "validation loss")
plt.title('Training  & validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()


# In[104]:


# Plot accuracy for each epoch in the training data 
plt.plot(results.epoch, results.history['acc'], "-b", label = "training accuracy")
plt.plot(results.epoch, results.history['val_acc'], "-r", label = "validation accuracy")
plt.title('Training  & validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()


# In[ ]:


# test on testing images 
for i in range(0, X_test.shape[0]):
    print(i)
    imshow(X_test[i], cmap='gray')
    plt.show()
    imshow((np.squeeze(preds_test_t[i])).astype(np.uint8))
    plt.show()

