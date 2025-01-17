
from DnD.src.lifted_ast import LiftedAST, AST_OP

class Weight:
    def __init__(self,dimensions,values,addrs):
        self.dimensions = dimensions
        self.values = values
        self.address = addrs

class Bias:
    def __init__(self,size,values,addrs):
        self.size = size
        self.values = values
        self.address = addrs

class Operator:
    def __init__(self,id,addr,ast,info):
        self.id = id
        self.addr = addr
        self.child = set()
        self.parent = None
        self.info = info
        self.weight = None
        self.bias = None
        self.layer = None
        self.ast = ast
        self.readbuffer = None
        self.writebuffer = None
        if self.ast is not None:
            self.readbuffer = ast.get_mem_read_base_reg(),
            self.writebuffer = ast.get_mem_write_base_reg(),
            if self.ast.op_type == AST_OP.CONV:
                dimension = [ 
                    self.ast.output_channel_iv.size(),
                    self.ast.input_channel_iv.size(),
                    self.ast.kernel_height_iv.size(),
                    self.ast.kernel_width_iv.size(),
                ]
                self.weight = Weight(dimension,self.info["weights"],self.info["weights addrs"])
                self.bias = Bias(info["output_channel"],info["bias"],info["bias addrs"])
            elif self.ast.op_type == AST_OP.FC:
                dimension = [self.ast.row_idx_iv.size(), self.ast.col_idx_iv.size()]
                self.weight = Weight(dimension,self.info["weights"],self.info["weights addrs"])
                self.bias = Bias(info["output_size"],info["bias"],info["bias addrs"])


    def getLayer(self):
        if self.layer is None:
            if self.parent is not None:
                self.layer = self.parent.getLayer() + 1
            else:
                self.layer = 1
        return self.layer

    #def getReadBuffer(self):
    #    print("getting read buffer for: ",self.id)
    #    if self.readbuffer[0] is None:
    #        print("parent: ",self.parent)
    #        if self.parent is not None:
    #            print("getting parents writebuffer")
    #            self.readbuffer = self.parent.getWritedBuffer()
    #    print("Read buffer of ",self.id," -> ",self.readbuffer)
    #    return self.readbuffer

    #def getWriteBuffer(self):
    #    print("getting writebuffer: ",self.id)
    #    if self.writebuffer[0] is None:
    #        c = list(self.child)[0]
    #        if c is not None:
    #            self.writebuffer = c.getReadBuffer()
    #    print("Write buffer of ",self.id," -> ",self.writebuffer)
    #    return self.writebuffer

class NewOp:
    def __init__(self,name,op,onnx_file,obj_file,wt_file, asm_file):
        self.name = name
        self.op = op
        self.onnxfile = onnx_file
        self.objfile = obj_file
        self.weightsfile = wt_file
        self.asmfile = asm_file
