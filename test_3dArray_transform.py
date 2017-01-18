import scipy.io
import numpy


images = scipy.io.loadmat("/home/stc/Programming/PycharmProjects/cnn_SAE_fuse/midData2.mat")
images = images['midData']

a = images[0, 0][3][0][0][4]
print(type(a))
print(a.shape)

# aa = numpy.split(a, 97,1)
# print(type(aa[0]))
# print(aa[0].shape)
aaa = numpy.stack(a, axis = 1)
print(type(aaa))
print(aaa.shape)

aa = a.T
print(type(aa))
print(aa.shape)

# oldData = [images[0, 0][3][0][0][4].T]

# for i in range(1, 98):
#     oldData.append(images[0, i][3][0][0][4].T)
#
# print(type(oldData))
# oldData = numpy.array(oldData)