#  http://blog.csdn.net/caanyee/article/details/52502759
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import numpy
import scipy.io
import random

# batch_size to train
batch_size = 4
# number of epochs to train
nb_epoch = 10
'''
    生成图像、数据预处理
'''
ad_num = 100
old_num = 98
#####################################################################################################
#####################################################################################################
random.seed(200)
#####################################################################################################
#####################################################################################################

images = scipy.io.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/oldData.mat")
images = images['oldData']

oldData = [images[0, 0][3][0][0][4]]
# oldData = [images[0, 0][3][0][0][4].T]
for i in range(1, 98):
    oldData.append(images[0, i][3][0][0][4])
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

adData = [images[0, 0][3][0][0][4]]
for i in range(1, 100):
    adData.append(images[0, i][3][0][0][4])

print(type(adData))
adData = numpy.array(adData)
adData -= adData.mean()
adData /= adData.max()
print(type(adData))
print(adData.shape)

del images

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
Y_train = numpy.concatenate(([[0, 1]] * 70, [[1, 0]] * 68))

X_test = numpy.concatenate((ad_test, old_test))
Y_test = numpy.concatenate(([[0, 1]] * 30, [[1, 0]] * 30))

print(type(X_train), type(Y_train), type(X_test), type(Y_test))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

input_tensor = Input(shape=(81, 97, 83))