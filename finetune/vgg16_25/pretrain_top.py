import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout
import random

'''
    搞数据
'''
adData = np.load('ad_features.npy')
# adData -= adData.mean()
# adData /= 255
oldData = np.load('old_features.npy')
# oldData -= oldData.mean()
# oldData /= 255


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
print(train_data.shape[1:])

model = Sequential()
model.add(BatchNormalization(input_shape=train_data.shape[1:]))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, nb_epoch=100, batch_size=10, validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')

