
# coding: utf-8

# In[ ]:

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
import cv2
import csv
from keras import backend as K
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
K.set_image_dim_ordering('tf')
os.environ['KERAS_BACKEND'] = "tensorflow"
print K.backend()


# In[ ]:

#Data Loading

img_path=[]
featureArray = []
labelArray=[]

img_path_test=[]
labelArray_test=[]
with open("list_attr_celeba.csv", "r") as ins:
    csv_reader = csv.reader(ins, delimiter=',')
    counter=0
    exitcounter=60
    noteCounter=1
    testCounter=30
    imgtolabel={}
    imgtolabel_test={}
    for row in csv_reader:
        if(counter==0):
            featureArray=row[1:]
        if(counter >= noteCounter):
            
            if(counter<testCounter):
                img_path.append(row[0])
                array=[]
                for x in row[1:]:
                    if(x=='1'):
                        array.append(1)
                    else:
                        array.append(0)
                imgtolabel[row[0]]=array
                labelArray.append(array)
                
            else:
                array=[]
                for x in row[1:]:
                    if(x=='1'):
                        array.append(1)
                    else:
                        array.append(0)
                img_path_test.append(row[0])
                labelArray_test.append(array)
                imgtolabel_test[row[0]]=array
        counter += 1

labelArray=np.array(labelArray)
labelArray_test=np.array(labelArray_test)


# In[ ]:

batch_size = 20
num_classes = 40
epochs = 12
feature_shape=(227,227,1)


# In[ ]:

def get_output(path,label_file=None):
    
   
    labels = label_file[path]
    
    return(labels)


# In[ ]:

def preprocess_input(image):
    #resize image
    newimg = cv2.resize(image,(227,227))
    return(newimg)


# In[ ]:

def get_input(path):
    path="/img_align_celeba/"+path
    img = cv2.imread(path,0)
    
    
    return(img)


# In[ ]:


def focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


# In[ ]:

def image_generator(files,label_file, batch_size = 80):
    
    while True:
          # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a = files, 
                                         size = batch_size)
        batch_input = []
        batch_output = [] 

          
          # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = get_input(input_path )
            output = get_output(input_path,label_file=label_file )
            
            input = preprocess_input(image=input)
            batch_input += [ input ]
            batch_output += [ output ]
          # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_x= batch_x.reshape(batch_size,227,227,1)
        batch_y = np.array( batch_output )
       
        
        yield( batch_x, batch_y )


# In[ ]:

model = Sequential()
#convulution 1

model.add(Conv2D(75,kernel_size=(7,7),strides=4,data_format='channels_last',activation='relu', input_shape=feature_shape))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization())



#convolution 2

model.add(Conv2D(200,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(BatchNormalization())

#convolution 3
model.add(Conv2D(300,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=2))
model.add(BatchNormalization())

#flattening
model.add(Flatten())
#FC1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#FC2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#FC3
model.add(Dense(num_classes, activation='relu'))

print model.summary()


# In[ ]:

# fitting generator
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.compile(loss=focal_loss_fixed,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit_generator(image_generator(img_path,imgtolabel, batch_size = 20),callbacks=callbacks_list,samples_per_epoch=50, nb_epoch=10)
model.save_weights('weights.h5')


# In[ ]:




# In[ ]:



