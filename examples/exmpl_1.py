
#In this example, we patch the MNIST dnn such that the patched binary will
#always misclassify 9 as 1.
#Specifically, we change the bias of the FC operator in the very last layer of
#the DNN as shown in the below code.

import dnnpatcher
from dnnpatcher import loadDNN

dnn = loadDNN("~/dnnpatcher/DnD/binary_samples/evkbimxrt1050_glow_lenet_mnist_release.axf")

# dnn.display() --display all the OPs with their ID

b = dnn.getBias(5)

v = b.values

v[9] = -10000
v[1] = 10000

dnn.changeBias(5,v)
dnn.applyPatches()
