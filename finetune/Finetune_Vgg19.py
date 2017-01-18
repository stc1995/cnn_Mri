#  http://blog.csdn.net/caanyee/article/details/52502759
from keras.applications.vgg19 import VGG19
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

# input_tensor = Input(shape=(97, 83, 81))
#ValueError: NumpyArrayIterator is set to use the dimension ordering convention "tf" (channels on axis 3), i.e. expected either 1, 3 or 4 channels on axis 3. However, it was passed an array with shape (138, 81, 97, 83) (83 channels).
#We've got an error while stopping in post-mortem: <class 'KeyboardInterrupt'>
# 只能对二维的图片（1\3\4通道）进行图片生成，用tensor假装80多通道图片不行
# # this could also be the output a different Keras model or layer
# input_tensor = Input(shape=(97, 83, 81))  # this assumes K.image_dim_ordering() == 'tf'
# # prepare data augmentation configuration
# train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#
# test_datagen = ImageDataGenerator()
#
#
# train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
#
# test_generator = train_datagen.flow(X_test, Y_test, batch_size=32)

'''
    选取模型的一部分
'''
# create the base pre-trained model
base_model = VGG19(input_shape=(97, 83, 81), weights='imagenet', include_top=False)

'''
    训练自己的top_layer,并fit之,得出顶层的权重
'''
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
# model = Model(input=base_model.input, output=predictions)
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9), loss='binary_crossentropy')

# train the model on the new data for a few epochs
# hist = model.fit_generator(train_generator, samples_per_epoch=5, nb_epoch=10, validation_data=test_generator, nb_val_samples=5)
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                 validation_data=(X_test, Y_test))
print(hist.history)

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


'''
    加上最后一个卷基层，一起fit
'''
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# we train our model again (this time fine-tuning the top 1 convolution blocks
# alongside the top Dense layers
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                 validation_data=(X_test, Y_test))
print(hist.history)

score = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
print(score)


