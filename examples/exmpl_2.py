
#In this example, we test dnnpatcher's ability to add new operators.
#In order to test that the new assembly patched on to the DNN binary is
#functionally correct, we add a new operator in such a way that it does not
#affect the overall functionality of the DNN and produces the same output as the
#original one.

# In MNIST model, we add a RELU operator after the first CONV operator. Note
# that the first CONV operator has a RELU already integrated in it. Therefore,
# adding a new RELU is mathematically NULL operation and does not affect the
# clasification results.
# The patched MNIST binary produces the same result as the original one.
#
# API: dnn.createNewOp(
#           "Op_type", [ list of predecessor OP IDs], 
#           [ list of successor OP IDs], 
#           { "attribute name", val} -- This argument is optional
#       )
#

import dnnpatcher
from dnnpatcher import loadDNN

dnn = loadDNN("~/dnnpatcher/DnD/binary_samples/evkbimxrt1050_glow_lenet_mnist_release.axf","IMXRT1050")

# dnn.display() --display all the OPs with their ID

dnn.createNewOp("Relu", [1], [2])
