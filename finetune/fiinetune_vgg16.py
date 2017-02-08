from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
import random

img_path = 'old_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
temp = image.img_to_array(img)
temp = np.expand_dims(temp, axis=0)
temp = preprocess_input(temp)
oldData = [temp]
for i in range(1, 98):
    img_path = 'old_%d.jpg' % i
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    temp = image.img_to_array(img)
    temp = np.expand_dims(temp, axis=0)
    temp = preprocess_input(temp)
    oldData.append(temp)

oldData = np.array(oldData)
print(oldData.shape) #(98, 1, 7, 7, 512)
# np.save('old_features.npy', features)

img_path = 'ad_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
temp = image.img_to_array(img)
temp = np.expand_dims(temp, axis=0)
temp = preprocess_input(temp)
adData = [temp]
for i in range(1, 100):
    img_path = 'ad_%d.jpg' % i
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    temp = image.img_to_array(img)
    temp = np.expand_dims(temp, axis=0)
    temp = preprocess_input(temp)
    adData.append(temp)

adData = np.array(adData)
print(adData.shape)       #(100, 1, 7, 7, 512)
# np.save('ad_features.npy', features)


ad_num = 100
old_num = 98
seed = 10
#####################################################################################################
#####################################################################################################
random.seed(seed)
#####################################################################################################
######################## #############################################################################
temp = list(range(ad_num))
random.shuffle(temp)
ad_train_num = temp[0: int(ad_num * 0.7)]
ad_test_num = temp[int(ad_num * 0.7):]
ad_train = []
ad_test = []
for i in ad_train_num:
    ad_train.append(adData[i])
for j in ad_test_num:
    ad_test.append(adData[j])
ad_train = np.array(ad_train)
ad_test = np.array(ad_test)

temp = list(range(old_num))
random.shuffle(temp)
old_train_num = temp[0: int(old_num * 0.7)]
old_test_num = temp[int(old_num * 0.7):]
old_train = []
old_test = []
for i in old_train_num:
    old_train.append(oldData[i])
for j in old_test_num:
    old_test.append(oldData[j])
old_train = np.array(old_train)
old_test = np.array(old_test)

train_data = np.concatenate((ad_train, old_train))
validation_data = np.concatenate((ad_test, old_test))



# the features were saved in order, so recreating the labels is easy
train_labels = np.concatenate(([[0, 1]] * 70, [[1, 0]] * 68))
validation_labels = np.concatenate(([[0, 1]] * 30, [[1, 0]] * 30))


base_model= Sequential()
base_model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

base_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
base_model.load_weights('/home/stc/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


top_model = Sequential()
top_model.add(BatchNormalization(input_shape=(1, 7, 7, 512)))
top_model.add(Flatten())
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights('bottleneck_fc_model.h5')

# add the model on top of the convolutional base

base_model.add(top_model)

# base_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# base_model.fit(train_data, train_labels, nb_epoch=100, batch_size=10, validation_data=(validation_data, validation_labels))
