
import angr
from collections import defaultdict
import pickle
import sys
import os
import capstone
import dnnop
import binary
import imxrt1050
from imxrt1050 import ImxRT1050
from binary import Binary
from dnnop import Operator, Weight, Bias, NewOp

from DnD.src.loader import load
from DnD.src.iv_identify import identify_iv
from DnD.src.ast import extract_ast
from DnD.src.lifter import lift
from DnD.src.lifted_ast import LiftedAST, AST_OP
from DnD.src.timeout import timeout
from DnD.src.onnx_builder import export_onnx, make_conv, make_relu, make_add
from DnD.decompiler import lift_func_to_ast, recover_topology

from patcherex2 import *
from patcherex2.targets import ElfArmMimxrt1052
import struct
import logging
import numpy as np
from collections import OrderedDict

import onnx
from onnx import helper, shape_inference
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx import AttributeProto, TensorProto, GraphProto, numpy_helper
import numpy as np
from onnx.checker import check_model, check_graph

from enum import Enum


#class POS(Enum):
#    BEFORE = 1
#    AFTER = 2


class Dnn:
    def __init__(self,bin_path,proj,isa_type,adj_map,lifted_ast_map,op_info):
        self.bin_path = bin_path
        self.op_map = {}
        self.op_addr_map = {}
        self.layer_map = defaultdict(set)
        self.op_id_ctr = 1
        self.new_op_addr = 1
        #self.proj = proj
        self.binary = Binary(bin_path,proj,isa_type)
        self.new_op = None
        for node,children in adj_map.items():
            op = None
            if node in self.op_addr_map:
                op = self.op_addr_map[node]
            else:
                op = Operator(self.op_id_ctr,node,lifted_ast_map[node],op_info[node])
                self.op_map[self.op_id_ctr] = op
                self.op_addr_map[node] = op
                self.op_id_ctr += 1

            for child in children:
                child_op = None
                if child in self.op_addr_map:
                    child_op = self.op_addr_map[child]
                else:
                    child_op = Operator(self.op_id_ctr,child,lifted_ast_map[child],op_info[child])
                    self.op_map[self.op_id_ctr] = child_op
                    self.op_addr_map[child] = child_op
                    self.op_id_ctr += 1
                op.child.add(child_op)
                child_op.parent = op
                #set the read/write buffer is DnD returns None

                if child_op.readbuffer[0] is None:
                    child_op.readbuffer = child_op.parent.writebuffer

                if child_op.writebuffer[0] is None:
                    if child_op.info["op"] == AST_OP.MAXPOOL or child_op.info["op"] == AST_OP.RELU:
                        in_w = child_op.parent.info["output_width"]
                        in_h = child_op.parent.info["output_height"]
                        in_c = child_op.parent.info["output_channel"]

                        in_size = in_c * in_h * in_w * 4
                        child_op.writebuffer = ( 
                            [ child_op.readbuffer[0][0] + in_size ],
                            [ None ] 
                        )


                child_id = child_op.id
                print("Child: ", child_id, " parent: ",self.op_map[child_id].parent.id)

        for id,op in self.op_map.items():
            layer = op.getLayer()
            self.layer_map[layer].add(id)

        self.op_type_map = {
            "Conv": AST_OP.CONV,
            "FC": AST_OP.FC,
            "Relu": AST_OP.RELU,
            "MaxPool": AST_OP.MAXPOOL,
            "Add": AST_OP.ADD,
            "AveragePool": AST_OP.AVGPOOL,
            "Softmax": AST_OP.SOFTMAX
        }

        
    def display(self):
        print("layer\top id\top type")
        lmap = OrderedDict(sorted(self.layer_map.items()))
        for layer,operators in lmap.items():
            for op_id in operators:
                op = self.op_map[op_id]
                print(layer,"\t",op_id,"\t",op.info["op"])

    def getWeights(self,op_id):
        assert op_id in self.op_map
        op = self.op_map[op_id]
        return op.weight

    def getBias(self,op_id):
        assert op_id in self.op_map
        op = self.op_map[op_id]
        return op.bias

    def getAttributes(self,op_id,attr):
        assert op_id in self.op_map
        op = self.op_map[op_id]
        info = op.info
        if attr in info:
            return info[attr]
        else:
            print("Invalid attribute name...below are the valid attributes for operator ",op.ast.op_type)
            print(op.info.keys())
        return None
    

    def createOnnxModel(self, op, model_name):

        created_nodes = []
        created_inputs = []
        created_outputs = []
        created_inits = []

        output_node = None
        # create input node
        inputs_node = make_tensor_value_info(
            "inputs",
            TensorProto.FLOAT,
            [
                1,  # batch size
                op.info["input_channel"],
                op.info["input_height"],
                op.info["input_width"],
            ],
        )
        if op.info["op"] == AST_OP.ADD:
            created_inputs.append(inputs_node)
            created_inputs.append(inputs_node)
        else:
            created_inputs.append(inputs_node)
        if op.info["op"] == AST_OP.CONV:

            node_name = "conv"

            # node
            nodes, inputs, inits = make_conv(
                node_name,
                "inputs",
                node_name + "_output",
                op.info["input_width"],
                op.info["input_height"],
                1,
                op.info["input_channel"],
                op.info["output_channel"],
                op.info["kernel_width"],
                op.info["kernel_height"],
                op.info["padding"],
                op.info["striding"],
                op.info["weights"],
                op.info["bias"],
            )
            created_nodes.extend(nodes)
            created_inits.extend(inits)
            conv_output = nodes[0].output[0]
            output_node = make_tensor_value_info(
                conv_output, 
                TensorProto.FLOAT, 
                [
                    1,  # batch size
                    op.info["output_channel"],
                    op.info["output_height"],
                    op.info["output_width"],
                ]
            )
            #created_outputs.append(output_node)
            #created_inputs.append(prev_output_node)

            if "relu" in op.info.keys() and op.info["relu"]:
                node_name = "relu"
                nodes = make_relu(node_name, conv_output, node_name + "_output")
                created_nodes.extend(nodes)
                relu_output_node = make_tensor_value_info(
                    nodes[0].output[0], 
                    TensorProto.FLOAT, 
                    [
                        1,  # batch size
                        op.info["output_channel"],
                        op.info["output_height"],
                        op.info["output_width"],
                    ]
                )
                output_node = relu_output_node

        elif op.info["op"] == AST_OP.RELU:

            node_name = "relu"
            nodes = make_relu(node_name, "inputs", node_name + "_output")
            created_nodes.extend(nodes)
            output_node = make_tensor_value_info(
                nodes[0].output[0],
                TensorProto.FLOAT, 
                [
                    1,  # batch size
                    op.info["output_channel"],
                    op.info["output_height"],
                    op.info["output_width"],
                ]
            )

        elif op.info["op"] == AST_OP.ADD:

            node_name = "add"
            nodes = make_add(
                node_name,
                "inputs",
                "inputs",
                node_name + "_output"
            )
            created_nodes.extend(nodes)
            output_node = make_tensor_value_info(
                nodes[0].output[0],
                TensorProto.FLOAT, 
                [
                    1,  # batch size
                    op.info["output_channel"],
                    op.info["output_height"],
                    op.info["output_width"],
                ]
            )

        else:
            assert False


        created_outputs.append(output_node)
        #print("Nodes: ",created_nodes)
        #print("inputs: ",created_inputs)
        #print("outputs: ",created_outputs)
        graph = make_graph(
            created_nodes, "test", created_inputs, created_outputs, created_inits
        )

        check_graph(graph)
        print("pass graph-check")

        model = helper.make_model(graph, producer_name="onnx-builder")
        #model = shape_inference.infer_shapes(model)
        check_model(model)
        print("pass model-check")
        print(model)

        onnx.save_model(model, model_name)



    def createNewOp(
            self, new_op_type_str, predecessor_lst = [], 
            successor_lst = [], new_op_attr = {}
        ):
        if new_op_type_str not in self.op_type_map:
            print("Invalid operator type: ",new_op_type_str)
            print("Valid operator names: ",self.op_type_map.keys())
            assert False
        if len(predecessor_lst) == 0 or len(successor_lst) == 0:
            print("Support for adding OP at the beginning or end of DNN is not supported yet")
            print("please supply predecessor and successor")
            assert False

        parent_op = self.op_map[predecessor_lst[0]]
        child_op = self.op_map[successor_lst[0]]

        if child_op not in parent_op.child:
            print("Predecessor and successor are not adjacent. Cannot add a node")
            assert False

        new_op_type = self.op_type_map[new_op_type_str]
        required_attr = {}
        model_name = "tmp/" + new_op_type_str + ".onnx"

        if new_op_type == AST_OP.CONV:

            assert len(new_op_attr) > 0
            new_op_attr["striding"] = 1
            new_op_attr["padding"] = 1
            new_op_attr["kernel_height"] = 3
            new_op_attr["kernel_width"] = 3
            print("Conv: enforcing input/output channel to be same as predecessor output_channel: ", parent_op.info["output_channel"])
            new_op_attr["input_channel"] = parent_op.info["output_channel"]
            new_op_attr["output_channel"] = parent_op.info["output_channel"]

            #if "striding" not in new_op_attr.keys():
            #    new_op_attr["striding"] = 1
            #if "padding" not in new_op_attr.keys():
            #    new_op_attr["padding"] = 1

            #if "kernel_height" not in new_op_attr.keys():
            #    print("Attribute 'kernel_height' not provided...aborting")
            #    return None
            #if "kernel_width" not in new_op_attr.keys():
            #    print("Attribute 'kernel_width' not provided...aborting")
            #    return None
            if "bias" not in new_op_attr.keys():
                print("Attribute 'bias' not provided...aborting")
                return None
            new_bias = new_op_attr["bias"]
            if len(new_bias) != new_op_attr["output_channel"]:
                print("Bias size is not equal to output channel ", new_op_attr["output_channel"])
                return None
            if "weights" not in new_op_attr.keys():
                print("Attribute 'weights' not provided...aborting")
                return None

            new_weights = new_op_attr["weights"]
            dims = new_weights.shape
            expected_dims = [
                new_op_attr["input_channel"],
                new_op_attr["output_channel"],
                new_op_attr["kernel_height"],
                new_op_attr["kernel_width"],
            ]
            for i in range(0,4):
                if dims[i] != expected_dims[i]:
                    print("Weights dimension mismatch!!!!")
                    dim_name = "output channel"
                    if i == 1:
                        dim_name = "input channel"
                    elif i == 2:
                        dim_name = "kernel height"
                    elif i == 3:
                        dim_name = "kernel width"
                    print("Dimension ", i," must be equal to ", dim_name, expected_dims[i])
                    return None
                #assert dims[i] >= 0 and dims[i] == w.dimensions[i]
            

        elif new_op_type == AST_OP.MAXPOOL or new_op_type == AST_OP.AVGPOOL or new_op_type == AST_OP.FC:
            print(
                "This operator changes output dimensions and cannot be applied using 'add operator API'. Use the API to add new layer"
            )
            assert False
        else:
            new_op_attr["input_channel"] = parent_op.info["output_channel"]
            new_op_attr["output_channel"] = parent_op.info["output_channel"]

        new_op_attr["input_height"] = parent_op.info["output_height"]
        new_op_attr["input_width"] = parent_op.info["output_width"]
        new_op_attr["output_width"] = parent_op.info["output_width"] 
        new_op_attr["output_height"] = parent_op.info["output_width"] 
        new_op_attr["op"] = new_op_type
        op = Operator(
            self.op_id_ctr, 
            -1 * self.new_op_addr,
            None,
            new_op_attr
        )
        self.op_id_ctr += 1
        self.new_op_addr += 1

        if op is not None:
            self.createOnnxModel(op, model_name)
            print("Generated ONNX: ",model_name)
            op.parent = parent_op
            for id in successor_lst:
                if id in self.op_map.keys():
                    op.child.add(self.op_map[id])
            self.new_op = NewOp(
                new_op_type_str,op,model_name,
                "tmp/" + new_op_type_str + ".o",
                "tmp/" + new_op_type_str + ".weights.bin",
                "tmp/" + new_op_type_str + ".s"
            )
        else:
            assert False

        self.binary.addNewOp(self.new_op)
        return model_name


    def pack_floats(self,array):
        flat_array = array.flatten()
        return struct.pack(f'{len(flat_array)}f', *flat_array)

    def changeWeight(self,op_id,weight):
        dims = weight.shape
        w = self.getWeights(op_id)
        assert len(dims) == len(w.dimensions)
        for i in range(0,len(w.dimensions)):
            assert dims[i] >= 0 and dims[i] == w.dimensions[i]
        addrs = w.address[0,0,0,0].item()
        print("patching new weights at: ",hex(addrs))
        packed_bytes = self.pack_floats(weight)
        self.binary.changeData(addrs, packed_bytes)


    def changeBias(self,op_id,bias):
        dims = len(bias)
        b = self.getBias(op_id)
        assert dims == b.size
        addrs = b.address[0].item()
        print("patching new bias at: ",hex(addrs))
        packed_bytes = self.pack_floats(bias)
        self.binary.changeData(addrs, packed_bytes)

def loadDNN(path,isa_type):
    # mnist sample
    # bin_path = "./binary_samples/mnist/evkbimxrt1050_glow_lenet_mnist_release.axf"

    isa = None
    if isa_type == "IMXRT1050":
        isa = ImxRT1050()
    else:
        print("Unknown architecture...aborting!!\nOnly IMXRT1050 is supported")
        return None

    proj = load(path)

    # AST
    lifted_ast_map = {}
    for f in proj.analysis_funcs:
        if f not in lifted_ast_map:
            lifted_ast = lift_func_to_ast(proj, f)
            if lifted_ast:
                lifted_ast_map[f] = lifted_ast

    # recover the info necessary for topology recovery
    for ast in lifted_ast_map.values():
        ast.recover()

    # recover topology
    adj_map = recover_topology(proj, lifted_ast_map)

    # recover attributes and weights
    op_info = {}
    state = proj.factory.blank_state()
    for ast_addr, ast in lifted_ast_map.items():
        prev_info = [
            op_info[addr] for addr in adj_map.keys() if ast_addr in adj_map[addr]
        ]

        info = ast.recover_attributes(prev_info)
        info["op"] = ast.op_type
        weights, bias, weights_addr, bias_addr = ast.extract_weights(state)
        info["weights"] = weights
        info["bias"] = bias
        info["weights addrs"] = weights_addr
        info["bias addrs"] = bias_addr
        op_info[ast_addr] = info

    dnn = Dnn(path,proj,isa,adj_map,lifted_ast_map,op_info)
    return dnn
