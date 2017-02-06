import numpy as np
import scipy.misc
import scipy.io

images = scipy.io.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/oldData.mat")
images = images['oldData']
x = images[0, 1][3][0][0][4]


x = np.random.random((600,800,3))
# scipy.misc.imsave("/home/stc/1.jpg",x)
scipy.misc.imsave("1.jpg",x)