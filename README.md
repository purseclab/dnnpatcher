# DNNPATCHER: A platform to patch deep neural network (DNN) binaries.


### Environment  
`git clone https://github.com/purseclab/DnD.git
 python3 -m venv patcher
 source patcher/bin/activate
 pip install -r ./DnD/angr_env/requirements.txt
 cp DnD/angr_env/base.py patcher/lib/python3.8/site-packages/claripy/ast/base.py
 pip install -U patcherex2
 deactivate `
