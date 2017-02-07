import scipy.io as sio
import scipy.misc as smi
import numpy as np

print("read start")
images = sio.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/oldData.mat")
print("read over")
images = images['oldData']

# for i in range(images[0, 1][3][0][0][4].shape[2]):
#     print(i)
#     x = images[0, 1][3][0][0][4][:,:,i]
#     name = "No:%s.jpg" % i
#     print(name)
#     smi.imsave(name, x)

slice = list(range(images[0, 1][3][0][0][4].shape[2]))
slice = slice[5:62:7]
total = np.array([])
for i in range(images.shape[1]):
    temp1 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[0]], images[0, i][3][0][0][4][:, :, slice[1]],
                            images[0, i][3][0][0][4][:, :, slice[2]]), axis=0)
    temp2 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[3]], images[0, i][3][0][0][4][:, :, slice[4]],
                            images[0, i][3][0][0][4][:, :, slice[5]]), axis=0)
    temp3 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[6]], images[0, i][3][0][0][4][:, :, slice[7]],
                            images[0, i][3][0][0][4][:, :, slice[8]]), axis=0)
    temp = np.concatenate((temp1, temp2, temp3), axis=1)
    total = np.concatenate(())
    name = "old_%d.jpg" % i
    smi.imsave(name, total)


print("read start")
images = sio.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/adData.mat")
print("read over")
images = images['adData']

# for i in range(images[0, 1][3][0][0][4].shape[2]):
#     print(i)
#     x = images[0, 1][3][0][0][4][:,:,i]
#     name = "No:%s.jpg" % i
#     print(name)
#     smi.imsave(name, x)

slice = list(range(images[0, 1][3][0][0][4].shape[2]))
slice = slice[5:62:7]
for i in range(images.shape[1]):
    temp1 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[0]], images[0, i][3][0][0][4][:, :, slice[1]],
                            images[0, i][3][0][0][4][:, :, slice[2]]), axis=0)
    temp2 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[3]], images[0, i][3][0][0][4][:, :, slice[4]],
                            images[0, i][3][0][0][4][:, :, slice[5]]), axis=0)
    temp3 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[6]], images[0, i][3][0][0][4][:, :, slice[7]],
                            images[0, i][3][0][0][4][:, :, slice[8]]), axis=0)
    total = np.concatenate((temp1, temp2, temp3), axis=1)
    name = "ad_%d.jpg" % i
    smi.imsave(name, total)
