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
slice = slice[3:76:3]
temp1 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[0]], images[0, 0][3][0][0][4][:, :, slice[1]],
                       images[0, 0][3][0][0][4][:, :, slice[2]], images[0, 0][3][0][0][4][:, :, slice[3]],
                       images[0, 0][3][0][0][4][:, :, slice[4]]), axis=0)
temp2 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[5]], images[0, 0][3][0][0][4][:, :, slice[6]],
                       images[0, 0][3][0][0][4][:, :, slice[7]], images[0, 0][3][0][0][4][:, :, slice[8]],
                       images[0, 0][3][0][0][4][:, :, slice[9]]), axis=0)
temp3 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[10]], images[0, 0][3][0][0][4][:, :, slice[11]],
                       images[0, 0][3][0][0][4][:, :, slice[12]], images[0, 0][3][0][0][4][:, :, slice[13]],
                       images[0, 0][3][0][0][4][:, :, slice[14]]), axis=0)
temp4 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[15]], images[0, 0][3][0][0][4][:, :, slice[16]],
                       images[0, 0][3][0][0][4][:, :, slice[17]], images[0, 0][3][0][0][4][:, :, slice[18]],
                       images[0, 0][3][0][0][4][:, :, slice[19]]), axis=0)
temp5 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[20]], images[0, 0][3][0][0][4][:, :, slice[21]],
                       images[0, 0][3][0][0][4][:, :, slice[22]], images[0, 0][3][0][0][4][:, :, slice[23]],
                       images[0, 0][3][0][0][4][:, :, slice[24]]), axis=0)
temp = np.concatenate((temp1, temp2, temp3, temp4, temp5), axis=1)
smi.imsave("old_0.jpg", temp)
total = [temp]
for i in range(1, images.shape[1]):
    temp1 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[0]], images[0, i][3][0][0][4][:, :, slice[1]],
                            images[0, i][3][0][0][4][:, :, slice[2]], images[0, i][3][0][0][4][:, :, slice[3]],
                            images[0, i][3][0][0][4][:, :, slice[4]]), axis=0)
    temp2 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[5]], images[0, i][3][0][0][4][:, :, slice[6]],
                           images[0, i][3][0][0][4][:, :, slice[7]], images[0, i][3][0][0][4][:, :, slice[8]],
                           images[0, i][3][0][0][4][:, :, slice[9]]), axis=0)
    temp3 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[10]], images[0, i][3][0][0][4][:, :, slice[11]],
                            images[0, i][3][0][0][4][:, :, slice[12]], images[0, i][3][0][0][4][:, :, slice[13]],
                           images[0, i][3][0][0][4][:, :, slice[14]]), axis=0)
    temp4 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[15]], images[0, i][3][0][0][4][:, :, slice[16]],
                           images[0, i][3][0][0][4][:, :, slice[17]], images[0, i][3][0][0][4][:, :, slice[18]],
                           images[0, i][3][0][0][4][:, :, slice[19]]), axis=0)
    temp5 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[20]], images[0, i][3][0][0][4][:, :, slice[21]],
                            images[0, i][3][0][0][4][:, :, slice[22]], images[0, i][3][0][0][4][:, :, slice[23]],
                           images[0, i][3][0][0][4][:, :, slice[24]]), axis=0)
    temp = np.concatenate((temp1, temp2, temp3, temp4, temp5), axis=1)
    total.append(temp)
    name = "old_%d.jpg" % i
    smi.imsave(name, temp)

total = np.array(total)
print(total.shape)
np.save('old_total.npy', total)



print("read start")
images = sio.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/adData.mat")
print("read over")
images = images['adData']
slice = list(range(images[0, 1][3][0][0][4].shape[2]))
slice = slice[3:76:3]
temp1 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[0]], images[0, 0][3][0][0][4][:, :, slice[1]],
                       images[0, 0][3][0][0][4][:, :, slice[2]], images[0, 0][3][0][0][4][:, :, slice[3]],
                       images[0, 0][3][0][0][4][:, :, slice[4]]), axis=0)
temp2 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[5]], images[0, 0][3][0][0][4][:, :, slice[6]],
                       images[0, 0][3][0][0][4][:, :, slice[7]], images[0, 0][3][0][0][4][:, :, slice[8]],
                       images[0, 0][3][0][0][4][:, :, slice[9]]), axis=0)
temp3 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[10]], images[0, 0][3][0][0][4][:, :, slice[11]],
                       images[0, 0][3][0][0][4][:, :, slice[12]], images[0, 0][3][0][0][4][:, :, slice[13]],
                       images[0, 0][3][0][0][4][:, :, slice[14]]), axis=0)
temp4 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[15]], images[0, 0][3][0][0][4][:, :, slice[16]],
                       images[0, 0][3][0][0][4][:, :, slice[17]], images[0, 0][3][0][0][4][:, :, slice[18]],
                       images[0, 0][3][0][0][4][:, :, slice[19]]), axis=0)
temp5 = np.concatenate((images[0, 0][3][0][0][4][:, :, slice[20]], images[0, 0][3][0][0][4][:, :, slice[21]],
                       images[0, 0][3][0][0][4][:, :, slice[22]], images[0, 0][3][0][0][4][:, :, slice[23]],
                       images[0, 0][3][0][0][4][:, :, slice[24]]), axis=0)
temp = np.concatenate((temp1, temp2, temp3, temp4, temp5), axis=1)
smi.imsave("ad_0.jpg", temp)
total = [temp]
for i in range(1, images.shape[1]):
    temp1 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[0]], images[0, i][3][0][0][4][:, :, slice[1]],
                            images[0, i][3][0][0][4][:, :, slice[2]], images[0, i][3][0][0][4][:, :, slice[3]],
                            images[0, i][3][0][0][4][:, :, slice[4]]), axis=0)
    temp2 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[5]], images[0, i][3][0][0][4][:, :, slice[6]],
                           images[0, i][3][0][0][4][:, :, slice[7]], images[0, i][3][0][0][4][:, :, slice[8]],
                           images[0, i][3][0][0][4][:, :, slice[9]]), axis=0)
    temp3 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[10]], images[0, i][3][0][0][4][:, :, slice[11]],
                            images[0, i][3][0][0][4][:, :, slice[12]], images[0, i][3][0][0][4][:, :, slice[13]],
                           images[0, i][3][0][0][4][:, :, slice[14]]), axis=0)
    temp4 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[15]], images[0, i][3][0][0][4][:, :, slice[16]],
                           images[0, i][3][0][0][4][:, :, slice[17]], images[0, i][3][0][0][4][:, :, slice[18]],
                           images[0, i][3][0][0][4][:, :, slice[19]]), axis=0)
    temp5 = np.concatenate((images[0, i][3][0][0][4][:, :, slice[20]], images[0, i][3][0][0][4][:, :, slice[21]],
                            images[0, i][3][0][0][4][:, :, slice[22]], images[0, i][3][0][0][4][:, :, slice[23]],
                           images[0, i][3][0][0][4][:, :, slice[24]]), axis=0)
    temp = np.concatenate((temp1, temp2, temp3, temp4, temp5), axis=1)
    total.append(temp)
    name = "ad_%d.jpg" % i
    smi.imsave(name, temp)

total = np.array(total)
print(total.shape)
np.save('ad_total.npy', total)
