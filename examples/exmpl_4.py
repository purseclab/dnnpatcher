#Adding new CONV operator
import numpy as np
import dnnpatcher
from dnnpatcher import loadDNN

dnn = loadDNN("~/dnnpatcher/DnD/binary_samples/evkbimxrt1050_glow_lenet_mnist_release.axf","IMXRT1050")

# dnn.display() --display all the OPs with their ID

new_conv_bias = np.random.rand(8).astype(np.float32)
new_conv_weights = np.random.rand(8,8,3,3).astype(np.float32)
new_op_attr = { "weights" : new_conv_weights, "bias" : new_conv_bias }

dnn.createNewOp("Conv",[1],[2],new_op_attr)
