
# coding: utf-8

# In[2]:

import os
import tensorflow.python.keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

import cv2
import glob
import pandas as pd
from skimage.io import imread_collection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

categories ={
	"airplane"   : 0,
	"automobile" : 1,
	"bird"       : 2,
	"cat"        : 3,
	"deer"       : 4,
	"dog"        : 5,
	"frog"       : 6,
	"horse"      : 7,
	"ship"       : 8,
	"truck"      : 9
}

print("\n\nReading and processing labels .....................")
new_data = {}
with open('E:/src/t_data/trainLabels.csv', 'r') as file:
	line = file.readline()
	while line:
		split_line = line.split(',')
		new_data[int(split_line[0])] = categories.get(split_line[1][:-1])
		line = file.readline()

# print ("THIS IS THE NEW DATA")
# print (new_data)

y_train = []
for key in new_data.keys():
	y_train.append(new_data.get(key))


y_train = np.array(y_train,dtype='float32')

#print (y_train)

y_train = utils.to_categorical(y_train,10)

print ("DONE processing lables ..................")

# print("999999999999999999999999999999999999999999999999999")
print("y_tarin_shape = ",y_train.shape)
# print (y_train)
# print("999999999999999999999999999999999999999999999999999")


#new_data[1]
#new_data.get(1)

print ("Reading and processing images .............")

cv_images = []
for img in glob.glob('E:/src/t_data/train/*.png'):
	n = cv2.imread(img)
	cv_images.append(n)

print ("Processed: ",len(cv_images), "images")
	
	
# for i in range (1,5):
	# print (cv_images[i])
	# plt.imshow(cv_images[i])
	# plt.show()

x_train = np.array(cv_images,dtype='float32')

print ("DONE with images .....................")
print("x_train_Shape = ",x_train.shape)


batch_size = 200

#Define the Model

cnn_model = Sequential()
cnn_model.add(Conv2D(64,kernel_size=(3,3),input_shape=(32,32,3), activation='relu'))
cnn_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling2D(pool_size=2))


cnn_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
cnn_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
#cnn_model.add(BatchNormalization())
#cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling2D(pool_size=2))

cnn_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
cnn_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
#cnn_model.add(MaxPooling2D(pool_size=2))


cnn_model.add(Flatten())

cnn_model.add(Dense(128, activation = 'relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(128, activation = 'relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(10, activation = 'softmax'))

cnn_model.summary()


# Compile the model

cnn_model.compile(
	loss = 'categorical_crossentropy',
    optimizer = Adam(lr=0.001),
    metrics = ['accuracy']
)


# Train the model

history = cnn_model.fit(x_train, y_train, validation_split=0.2,batch_size=batch_size, epochs=5, verbose=1)

print(history.history.keys())

 # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

 # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


cnn_model.save('model_1.h5')
#my_model=load_model(model_1.h5)

# results = model.predict(test_data)




