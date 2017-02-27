import scipy.io as sio
import scipy.misc as smi
import numpy as np
from keras.preprocessing import image

'''
    old
'''
print("read start")
images = sio.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/oldData.mat")
print("read over")
images = images['oldData']

# temp0 = images[0, 0][3][0][0][4][:, :, 0]
# temp1 = images[0, 0][3][0][0][4][:, :, 1]
# temp2 = images[0, 0][3][0][0][4][:, :, 2]
#
# temp = np.zeros((81, 97, 3))
# temp[:, :, 0] = temp0
# temp[:, :, 1] = temp1
# temp[:, :, 2] = temp2
# print(temp.shape)
# smi.imsave("old_0_temp0.jpg", temp)
# img = image.load_img("old_0_temp0.jpg", target_size=(224, 224))
# img_matrix = image.img_to_array(img)
total = np.zeros((224, 224, 81))
print(total.shape)
for i in range(images.shape[1]):
    for j in range(int(images[0, 0][3][0][0][4].shape[2]/3)):
        temp0 = images[0, i][3][0][0][4][:, :, 3*j]
        temp1 = images[0, i][3][0][0][4][:, :, 3*j+1]
        temp2 = images[0, i][3][0][0][4][:, :, 3*j+2]

        temp = np.zeros((81, 97, 3))
        temp[:, :, 0] = temp0
        temp[:, :, 1] = temp1
        temp[:, :, 2] = temp2

        # smi.imsave("./Material/old_%d_temp%d.jpg" % (i, j), temp)
        # img = image.load_img("./Material/old_%d_temp%d.jpg" % (i, j), target_size=(224, 224))
        smi.imsave("./Material/old_%d_temp.jpg" % i, temp)
        img = image.load_img("./Material/old_%d_temp.jpg" % i, target_size=(224, 224))
        img_matrix = image.img_to_array(img)
        total[:, :, 3*j] = img_matrix[:, :, 0]
        total[:, :, 3*j+1] = img_matrix[:, :, 1]
        total[:, :, 3*j+2] = img_matrix[:, :, 2]

    np.save('./Material/old_%d.npy' % i, total)

'''
    ad
'''
print("read start")
images = sio.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/adData.mat")
print("read over")
images = images['adData']

# temp0 = images[0, 0][3][0][0][4][:, :, 0]
# temp1 = images[0, 0][3][0][0][4][:, :, 1]
# temp2 = images[0, 0][3][0][0][4][:, :, 2]
#
# temp = np.zeros((81, 97, 3))
# temp[:, :, 0] = temp0
# temp[:, :, 1] = temp1
# temp[:, :, 2] = temp2
# print(temp.shape)
# smi.imsave("ad_0_temp0.jpg", temp)
# img = image.load_img("ad_0_temp0.jpg", target_size=(224, 224))
# img_matrix = image.img_to_array(img)
total = np.zeros((224, 224, 81))
print(total.shape)
for i in range(images.shape[1]):
    for j in range(int(images[0, 0][3][0][0][4].shape[2]/3)):
        temp0 = images[0, i][3][0][0][4][:, :, 3*j]
        temp1 = images[0, i][3][0][0][4][:, :, 3*j+1]
        temp2 = images[0, i][3][0][0][4][:, :, 3*j+2]

        temp = np.zeros((81, 97, 3))
        temp[:, :, 0] = temp0
        temp[:, :, 1] = temp1
        temp[:, :, 2] = temp2

        # smi.imsave("./Material/ad_%d_temp%d.jpg" % (i, j), temp)
        # img = image.load_img("./Material/ad_%d_temp%d.jpg" % (i, j), target_size=(224, 224))
        smi.imsave("./Material/ad_%d_temp.jpg" % i, temp)
        img = image.load_img("./Material/ad_%d_temp.jpg" % i, target_size=(224, 224))
        img_matrix = image.img_to_array(img)
        total[:, :, 3*j] = img_matrix[:, :, 0]
        total[:, :, 3*j+1] = img_matrix[:, :, 1]
        total[:, :, 3*j+2] = img_matrix[:, :, 2]

    np.save('./Material/ad_%d.npy' % i, total)