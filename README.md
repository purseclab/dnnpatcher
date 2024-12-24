# DNNPATCHER: A platform to patch deep neural network (DNN) binaries.

dnnpatcher is a framework for patching DNN binaries. The main goal of this
framework is to bridge the gap between low-level binary patching operations and
high-level DNN network modifications. It takes user requests such as changing
weights/bias of an operator or add a new operator and translates them to
low-level binary patching operations and applies them to generate a new DNN
binary. For example, to add a new operator, it accepts operator type, position
in the DNN and its attributes. Based on these user provided info, it first
generates a high-level ONNX model for this new operator, compiles this ONNX
model to assembly function and patches it on to DNN binary. Additionally, it
also changes the dispatcher function of the DNN to place a call to this new
operator at appropriate location on the DNN.

dnnpatcher has been built on top of DnD, a DNN decompiler and patcherex2,
a versatile binary patching framework.

### Environment 

Follow the below bash commands to prepare the enviornment for dnnpatcher.

```bash
git clone https://github.com/purseclab/DnD.git
python3 -m venv patcher
source patcher/bin/activate
pip install -r ./DnD/angr_env/requirements.txt
cp DnD/angr_env/base.py patcher/lib/python3.8/site-packages/claripy/ast/base.py
pip install -U patcherex2
```

dnnpatcher reliaes on GLOW compiler to compile new operaters for the
*createNewOp* API. The glow compiler must be placed ${HOME} and installed by
following directions mentioned in https://github.com/pytorch/glow.


### API and usage

dnnpatcher supports below dnn modification operations:

1. *changeWeights(op\_id, new\_weights):* Change weights of a given operator
   (op\_id).
2. *changeBias(op\_id, new\_weights):* Change bias of a given operator (op\_id).
3. *createNewOp("op_type", [ predecessor op_id list ], [ successor op_id list],
   { "attr name", val }):* Add a new operator between the given list of
   predecessor operators and successor operators.

Examples of how to use these dnn modification operations are present in
*examples/* directory. The operations can be implemented in an interactive
manner using ipython terminal as shown in the videos in *examples/* directory. 


