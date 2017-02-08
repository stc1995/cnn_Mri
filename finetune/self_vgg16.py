from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D

# def VGG_16(weights_path=None):
#     model = Sequential()
#     model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3, 224, 224)))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#     # model.add(Flatten())
#     # model.add(Dense(4096, activation='relu'))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(4096, activation='relu'))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(1000, activation='softmax'))
#
#     if weights_path:
#         model.load_weights(weights_path)
#
#     return model


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
print(1)
model = VGG_16('/home/stc/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
print(2)