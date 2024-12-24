# DNNPATCHER: A platform to patch deep neural network (DNN) binaries.


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

### API and usage

dnnpatcher supports below dnn modification operations:

1. *changeWeights(op\_id, new\_weights):* Change weights of a given operator
   (op\_id)
2. *changeBias(op\_id, new\_weights):* Change bias of a given operator (op\_id)
3. *createNewOp("op_type", [ predecessor op_id list ], [ successor op_id list],
   { "attr name", val }):* Add a new operator between the given list of
   predecessor operators and successor operators.
