# https://github.com/damayant/Detection-Of-Parkinson-s-Disease-with-Brain-MRI-using-Deep-Learning
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten

import numpy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
import csv
# SKLEARN
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split

#####################################################################################################################################
# input image dimensions
img_rows, img_cols = 81, 83

# number of channels
img_channels = 97

# data,Label = shuffle(immatrix,label, random_state=2)
# train_data = [data, Label]imp

# batch_size to train
batch_size = 4
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3



# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

'''
    数据预处理
'''
ad_num = 100
old_num = 98

images = scipy.io.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/oldData.mat")
images = images['oldData']

oldData = [numpy.stack(images[0, 0][3][0][0][4], axis=1)]
# oldData = [images[0, 0][3][0][0][4].T]
for i in range(1, 98):
    oldData.append(numpy.stack(images[0, i][3][0][0][4], axis=1))
    # oldData.append(images[0, i][3][0][0][4].T)

print(type(oldData))
oldData = numpy.array(oldData)
# oldData = oldData.astype('float32')
oldData -= oldData.mean()
oldData /= oldData.max()
print(type(oldData))
print(oldData.shape)

images = scipy.io.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/adData.mat")
images = images['adData']

adData = [numpy.stack(images[0, 0][3][0][0][4], axis=1)]
for i in range(1, 100):
    adData.append(numpy.stack(images[0, i][3][0][0][4], axis=1))
    # adData.append(images[0, i][3][0][0][4].T)

print(type(adData))
adData = numpy.array(adData)
adData -= adData.mean()
adData /= adData.max()
print(type(adData))
print(adData.shape)

csvfile = open('ad_old_multiW1b1_10.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(['seed', 'index of max acc', 'max acc of this seed'])
# f = open("logging.txt", "w")
for seed in range(0, 1000, 200):
    #####################################################################################################################################
    model = Sequential()

    model.add(Convolution2D(20, nb_conv, nb_conv, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2, 2)))  # (1,1)
    model.add(Dropout(0.5))

    model.add(Convolution2D(50, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=(2, 2)))  # (1,1)
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(600))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

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
    ad_train = numpy.array(ad_train)
    ad_test = numpy.array(ad_test)

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
    old_train = numpy.array(old_train)
    old_test = numpy.array(old_test)

    X_train = numpy.concatenate((ad_train, old_train))
    # theano后端
    # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
    Y_train = numpy.concatenate(([[0, 1]] * 70, [[1, 0]] * 68))

    X_test = numpy.concatenate((ad_test, old_test))
    # theano后端
    # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    Y_test = numpy.concatenate(([[0, 1]] * 30, [[1, 0]] * 30))

    print(type(X_train), type(Y_train), type(X_test), type(Y_test))
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                     validation_data=(X_test, Y_test))
    #####################################################################################################################################
    score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_acc']
    val_loss = numpy.array(val_loss)
    val_acc = numpy.array(val_acc)
    # print(type(val_loss), type(val_acc))
    # print(val_acc.argmax(), val_acc.max(), file=f)
    print(val_acc.argmax(), val_acc.max())
    data = [seed, val_acc.argmax(), val_acc.max()]
    writer.writerow(data)

    plt_name = 'MRI_DL_plot'+str(seed)+'.png'
    xc = range(nb_epoch)
    plt.figure(seed, figsize=(5, 5))
    plt.plot(xc, val_loss)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('acc')
    plt.title('val_loss and val_acc')
    plt.savefig(plt_name)
    plt.show()

    del val_loss
    del val_acc
    del model

csvfile.close()
# #####################################################################################################################################
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # saving weights
# fname = "version3.hdf5"
# model.save_weights(fname, overwrite=True)
