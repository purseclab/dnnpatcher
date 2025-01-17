
import angr
import capstone

class Assembly:
    def __init__(self,isa_type):
        self.isa = isa_type

    def isaType(self):
        return self.isa.isaType()

    def getCallTgtSym(self, asm_insn):
        words = asm_insn.split()
        if self.isa.isCallMne(words[0]):
            return words[1]
        return None

    def convertLabelLoadToPCRltv(self,asm):
        return self.isa.convertLabelLoadToPCRltv(asm)

    def getCallTgt(self, insn):
        target_address = None
        if self.isa.isCall(insn):
            if insn.operands:
                for operand in insn.operands:
                    target_address = operand.imm

        return target_address

    def saveGPR(self):
        return self.isa.pushGPR()

    def restoreGPR(self):
        return self.isa.popGPR()

    def call(self, tgt_sym):
        return self.isa.call(tgt_sym)

    def jump(self, tgt_sym):
        return self.isa.jump(tgt_sym)

    def byte(self, val):
        return self.isa.byte(val)

    def callArgConst(self, const, argnum):
        return self.isa.callArgConst(const, argnum)

    def callArgLabel(self, label, argnum):
        return self.isa.callArgLabel(label, argnum)
