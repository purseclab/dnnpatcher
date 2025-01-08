
import angr
import pickle
import sys
import os
import capstone
import assembly
import logging
import numpy as np
from assembly import Assembly
from patcherex2 import *
from patcherex2.targets import ElfArmMimxrt1052

class Binary:
    def __init__(self,bin_path,proj,isa_type):
        self.bin_path = bin_path
        self.proj = proj # Angr object
        self.assembly = Assembly(isa_type)
        self.patcher = None
        if self.assembly.isaType() == "EMXRT1050":
            logging.getLogger("patcherex").setLevel("DEBUG")
            self.patcher = Patcherex(self.bin_path, target_cls=ElfArmMimxrt1052)

    def applyPatches(self):
        file_name = os.path.splitext(self.bin_path)[0]
        ext = os.path.splitext(self.bin_path)[1]
        self.patcher.apply_patches()
        self.patcher.binfmt_tool.save_binary(f"{file_name}_patched{ext}")

    def changeData(self, addrs, packed_bytes):
        self.patcher.patches.append(ModifyDataPatch(addrs, packed_bytes))
        self.applyPatches()

    def extractCodeBytes(self,path_to_obj,target_sym):
        obj = angr.Project(path_to_obj, auto_load_libs=False)
        
        main_obj = obj.loader.main_object
        
        text_section = None
        for section in main_obj.sections:
            if section.name == '.text':
                text_section = section
                break
        
        symbol = obj.loader.find_symbol(target_sym)
        #mask = 0xfffffffe
        sym_addr = symbol.rebased_addr
        
        asm = "extracted_code:\n"
    
        if symbol:
            print(f"Symbol {target_sym} found!")
            print(f"Address: {hex(sym_addr)}")
        else:
            print(f"Symbol {target_sym} not found.")
        
        block = obj.factory.block(sym_addr)
        for insn in block.capstone.insns:
            print(f"0x{insn.address:x}: {insn.mnemonic} {insn.op_str}")
            if insn.mnemonic in ["bl", "blx"]:
                if insn.operands:
                    for operand in insn.operands:
                        target_address = operand.imm
                        print("Found target: ",hex(target_address))
                        sym_addr = target_address
        
        if text_section is None:
            print(".text section not found.")
        else:
            start_addr = text_section.vaddr
            size = text_section.memsize
        
            # Extract the raw bytes from the memory using angr's memory object
            text_bytes = obj.loader.memory.load(start_addr, size)
    
            end_addr = start_addr + size
            i = 0
            while start_addr < end_addr:
                if start_addr == sym_addr:
                    asm = asm + target_sym + ":\n"
                b = text_bytes[i];
                asm = asm + "  .byte " + str(b) + "\n"
                i += 1
                start_addr += 1
    
        #print(asm)
        return asm

    def addNewOpToDispatcher(self,new_op_sym,predecessor_op):
        dispatcher_func = self.proj.funcs[self.proj.dispatch_addr]
        #op_call_sites = {}
        if predecessor_op is None:
            assert False
            #Need to add implementation for OP added at the beginning of DNN
        dispatcher_sym = "new_dispatcher"
        asm = "new_dispatcher:\n";
        for block in dispatcher_func.blocks:
            for insn in block.capstone.insns:
                asm = asm + "  " + insn.mnemonic + " " + insn.op_str + "\n"
                target_address = self.assembly.getCallTgt(insn)
                if target_address is not None and target_address == predecessor_op.addr:
                    asm = asm + "  " + self.assembly.saveGPR() + "\n"
                    asm = asm + "  " + self.assembly.call(new_op_sym) + "\n"
                    asm = asm + "  " + self.assembly.restoreGPR() + "\n"

        return dispatcher_sym, asm

    def getDispatchCallSite(self):
        dispatcher_caller = self.proj.funcs[self.proj.dispatch_caller_addr]
        dispatch_call_site = None
        for block in dispatcher_caller.blocks:
            for insn in block.capstone.insns:
                target_address = self.assembly.getCallTgt(insn)
                if target_address is not None and target_address == self.proj.dispatch_addr:
                    dispatch_call_site = insn.address

        print("Dispatch call site: ",hex(dispatch_call_site))
        return dispatch_call_site

    def createCallSiteForNewOp(
            self,predecessor_op, successor_op, target_sym,
            weight_sym, bias_sym
        ):
        #currently on handle ARM and currently handling only input/output buffer
        #passing.
        #Needs to be improved to pass a generic list of arguments.

        input_buffer = predecessor_op.writebuffer[0]
        output_buffer = successor_op.readbuffer[0]

        if input_buffer is None and output_buffer is None:
            print("Input/Output buffer not available")
            assert False
        elif input_buffer is None:
            input_buffer = output_buffer
        elif output_buffer is None:
            output_buffer = input_buffer
       
        tramp_sym = target_sym + "_tramp"

        asm = tramp_sym + ":\n"
        asm = asm + self.assembly.callArgConst(input_buffer[0],1)
        asm = asm + self.assembly.callArgConst(output_buffer[0],2)
        asm = asm + self.assembly.jump(target_sym) + "\n"

        return tramp_sym,asm

    def addNewOp(self, new_op):
        cmd = (
            "~/glow/build_Debug/bin/model-compiler -backend=CPU -model=" + 
            new_op.onnxfile + 
            " -emit-bundle=tmp/ -target=arm -mcpu=cortex-m7"
        )
        os.system(cmd)
        op_asm = self.extractCodeBytes(new_op.objfile, new_op.name)
        tramp_sym, tramp_asm = self.createCallSiteForNewOp(
            new_op.op.parent, 
            list(new_op.op.child)[0], 
            new_op.name,
            None,None
        )
        dispatcher_sym, dispatcher_asm = self.addNewOpToDispatcher(
            tramp_sym,
            new_op.op.parent 
        )

        op_asm = dispatcher_asm + tramp_asm + op_asm

        #print(dispatcher_asm)
        print(op_asm)
        self.patcher.patches.append(
            InsertInstructionPatch("dispatch_new", op_asm, is_thumb=True)
        )
        dispatch_call_site = self.getDispatchCallSite()

        print("Dispatch call site: ",hex(dispatch_call_site))

        mask = 0xfffffffe
        self.patcher.patches.append(
            ModifyInstructionPatch(
                dispatch_call_site & mask, 
                self.assembly.call("{dispatch_new}")
            )
        )
        self.applyPatches()
