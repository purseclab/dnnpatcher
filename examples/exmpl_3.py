
# In the previous example (exmpl_2.py), we tested the functional correctness of
# the patched assembly code added by the createNewOp API. 
# In this example, we try to change the behavior of a DNN by adding a new
# operator.
#
# We change the RESNET DNN by adding a RELU operator between the CONV operator
# in 13th layer and the ADD operator in 14th layer. This resulted in
# a missclassification of an "airplane" to "ship".

import dnnpatcher
from dnnpatcher import loadDNN

dnn = loadDNN("~/dnnpatcher/DnD/binary_samples/evkbimxrt1050_glow_lenet_mnist_release.axf","IMXRT1050")

# dnn.display() --display all the OPs with their ID

dnn.createNewOp("Relu", [13], [14])
