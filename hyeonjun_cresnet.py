#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import matplotlib.ticker as plticker

#add savemodel 
import pandas as pd

#입력 매개변수 리스트 수정
#add fileter number flag
#입력 매개변수 리스트 수정 export
#0번째 블록의 resblock의 BN과정 삭제





# In[2]:



import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[3]:


def resnet_layer(inputs,
                 num_filters,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
   
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def conv1_layer(inputs,
                num_filters=64,
                kernel_size=7,
                strides=2, 
                activation='relu',
                batch_normalization=True, 
                conv_first=True):
    
    x = resnet_layer(inputs,
                 num_filters=num_filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True)
    
    '''
    first_conv = Conv2D(64,
                  kernel_size=(7, 7),
                  strides=(2, 2),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
  
    x = inputs
    x = conv1_layer(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
'''
    
    return x

def shortcut_layer(inputs, num_filters, isIdentity, isBatch_normalization, shortcut_strides):
    shortcut = inputs
    
    if isIdentity == True:
        shortcut = Conv2D(num_filters, (1, 1), strides=shortcut_strides, padding='valid')(shortcut)
    if isBatch_normalization == True:
        shortcut = BatchNormalization()(shortcut)
        
    return shortcut


# In[4]:


def resnet(input_shape, re=[4, [2,2,2,2], 2], num_classes=2, isNumFilers=True):
    
    name = f'{re}'
    
    num_filters = 64
    count = 0
    inputs = Input(shape=input_shape) #(224, 224, 3)
    print(inputs)
    
    '''x = resnet_layer(inputs,
                 num_filters,
                 kernel_size=7,
                 strides=2,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True)'''
    count+=1
    x = conv1_layer(inputs=inputs)
    print(x)
    
    x = MaxPooling2D((3, 3), 2, padding='same')(x) #(56, 56, 3)
    print(x)
    
    #shortcut = x
    
    for layer in range(re[0]): #4 0~3
        print('----------------')
        #shortcut = x
        for res_block in range(re[1][layer]):  #2 0~1
            shortcut = x
            
            print('----------------')
            for in_res_block in range(re[2]): # 2 0~1
                
                strides = 1
                if layer > 0 and res_block == 0 and in_res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                    
                isActivation = 'relu'
                if in_res_block == re[2]-1:
                    isActivation = None
                count+=1
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides,
                                activation=isActivation)
                
                print(f'{count}::{num_filters}X{num_filters},{strides} : {x}')

            '''if layer != 0 and res_block == 0:'''
            '''if res_block == 0:
                shortcut = Conv2D(num_filters, (1, 1), strides=shortcut_strides, padding='valid')(shortcut)
                shortcut = BatchNormalization()(shortcut)'''
            
            shortcut_strides=1
            if layer > 0 and res_block == 0:  # first layer but not first stack
                    shortcut_strides = 2  # downsample
            
            if layer == 0 and res_block == 0:
                shortcut = shortcut_layer(shortcut,
                                          num_filters,
                                          isIdentity=False,
                                          isBatch_normalization=False, 
                                          shortcut_strides=shortcut_strides)
                
            elif layer > 0 and res_block == 0:
                shortcut = shortcut_layer(shortcut, 
                                          num_filters,
                                          isIdentity=True,
                                          isBatch_normalization=True,
                                          shortcut_strides=shortcut_strides)
                
            print(shortcut)
            x = Add()([x, shortcut])
            print(x)
            x = Activation('relu')(x)
            print(x)
            
        if isNumFilers == True:
            num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    
    #x = AveragePooling2D(pool_size=8)(x)
    x = GlobalAveragePooling2D()(x)
    print(x)
    #y = Flatten()(x)
    count+=1
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, count, name


# In[5]:


def make_path(base_dir):
    dir_path = os.path.dirname(base_dir)
    file_name = os.path.basename(base_dir)
    full_path = os.path.join(dir_path, file_name)
    
    return dir_path, file_name, full_path


# In[6]:


def viewResult(history, base_dir):
    
    '''image_dir = os.path.dirname(base_dir)
    image_file = os.path.basename(base_dir) + '_image.png'
    full_path = os.path.join(image_dir, image_file)'''
    
    #model_base_dir = f'/data/hyeon_model_save/layer_image/{classes}'
    
    dir_path, file_name, full_path = make_path(base_dir+'_image.png')
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    #epochs = range(1, len(acc) + 1)
    epochs = [tick for tick in range(1, len(acc)+1)]

    fig = plt.figure(figsize=(10,4))
    plt.suptitle(f'{file_name}')
    plt.subplot(1,2,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    
    plt.plot(epochs, acc, 'b', label='Training acc', linestyle=':')
    plt.plot(epochs, val_acc, 'r', label='Validation acc', linestyle='-')

    
    plt.xticks = np.arange(min(acc), max(acc)+1, 0.1, dtype='f')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(b=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    
    plt.subplot(1,2,2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, loss, 'b', label='Training loss', linestyle=':')
    plt.plot(epochs, val_loss, 'r', label='Validation loss', linestyle="-")
    plt.title(f'Loss')
    plt.grid(b=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.yticks = np.arange(min(acc), max(acc)+1, 0.1, dtype='f')
    #fig.savefig(full_path)
    print(full_path)
    plt.show()
    


# In[19]:


def viewResult_v2(history, base_dir):

    dir_path, file_name, full_path = make_path(base_dir+'_image.png')
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise
            

 
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    '''acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']'''
    
    #epochs = range(1, len(acc) + 1)
    epochs = [tick for tick in range(1, len(acc)+1)]
    
    fig = plt.figure(figsize=(10,4))
    plt.suptitle(f'{file_name}')
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.set(title='Accuracy', xlabel='epoch', ylabel='accuracy')
    ax1.plot(epochs, acc, 'b', label='Training acc', linestyle=':')
    ax1.plot(epochs, val_acc, 'r', label='Validation acc', linestyle='-')
    ax1.grid(b=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, 0.05))
    ax1.yaxis.set_major_formatter(plticker.FormatStrFormatter('%0.2f'))
    ax1.xaxis.set_major_formatter(plticker.FormatStrFormatter('%0.1f'))
    ax2 = fig.add_subplot(1,2,2)
    ax2.set(title='Loss', xlabel='epoch', ylabel='loss')
    ax2.plot(epochs, loss, 'b', label='Training loss', linestyle=':')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss', linestyle="-")
    ax2.grid(b=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, 0.5))
    ax2.yaxis.set_major_formatter(plticker.FormatStrFormatter('%0.1f'))
    ax2.xaxis.set_major_formatter(plticker.FormatStrFormatter('%0.1f'))
    fig.savefig(full_path)
    print(full_path)
    plt.show()
    


# In[9]:


def save_result_image_list(model_list, history_list, base_dir):
    for index in range(len(history_list)):
        viewResult_v2(history_list[index],
                      base_dir+f"result_image/{model_list[index][1]}_{model_list[index][2]}")
    
    


# In[10]:


def save_model(model, base_dir):         
    dir_path, file_name, full_path = make_path(base_dir)
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise

    model_json = model.to_json()
    with open(full_path, "w") as json_file: 
        json_file.write(model_json)
    print(full_path)


# In[11]:


def save_model_list(model_list, base_dir):
    for model in model_list:
        save_model(model[0], base_dir+f"model/{model[1]}_{model[2]}_model.json")
        


# In[12]:


def save_weight(model, base_dir):
    dir_path, file_name, full_path = make_path(base_dir)
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise
            
    model.save_weights(full_path)
    print(full_path)  


# In[13]:


def save_weight_list(model_list, base_dir):
    for model in model_list:
        save_weight(model[0], base_dir+f"weight/{model[1]}_{model[2]}_weight.h5")
        


# In[14]:


def save_history(model_list, history_list, base_dir):
    dir_path, file_name, full_path = make_path(base_dir+'history/_history.txt')
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise

    with open(full_path, 'w') as f:
        for index in range(len(history_list)):
            f.write(f"{model_list[index][1]}_{model_list[index][2]}\n")
            f.write(f"acc\n{history_list[index].history['acc']}\n")
            f.write(f"loss\n{history_list[index].history['loss']}\n")
            f.write(f"val_acc\n{history_list[index].history['val_acc']}\n")          
            f.write(f"val_loss\n{history_list[index].history['val_loss']}\n") 
        print(full_path)


# In[15]:


def model_evaluate(model_list, test_generator, base_dir):
    dir_path, file_name, full_path = make_path(base_dir+'_evaluate.txt')
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise
    
    with open(full_path, 'w') as fh:
        for model in model_list:
            scores = model[0].evaluate_generator(test_generator, steps=test_generator.n/test_generator.batch_size)
            fh.write("[%s_%s] \t %.2f%%" %(model[1], model[2], scores[1]*100))
            print("[%s_%s] \t %.2f%%" %(model[1], model[2], scores[1]*100))
        


# In[16]:


def save_model_summary(model, base_dir):
    dir_path, file_name, full_path = make_path(base_dir)
    try:
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise
    
    with open(full_path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print(full_path)


# In[17]:


def save_model_summary_list(model_list, base_dir):
    for model in model_list:
        save_model_summary(model[0], base_dir+f"summary/{model[1]}_{model[2]}_summary.txt")
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




