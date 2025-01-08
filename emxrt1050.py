
import angr
import capstone

class EmxRT1050:

    def isaType(self):
        return "EMXRT1050"

    def isCall(self,insn):
        if insn.mnemonic in ["bl", "blx"]:
            return True
        return False

    def pushGPR(self):
        asm = "  push {R0-R12}"
        return asm

    def popGPR(self):
        asm = "  pop {R0-R12}"
        return asm

    def call(self, tgt_sym):
        asm = "  bl " + tgt_sym
        return asm

    def jump(self, tgt_sym):
        asm = "  b " + tgt_sym
        return asm

    def argReg(self, argnum):
        arg_reg = "R0"
        if argnum == 2:
            arg_reg = "R1"
        elif argnum == 3:
            arg_reg = "R2"
        elif argnum == 3:
            arg_reg = "R3"

        return arg_reg

    def callArgConst(self, const, argnum):

        arg_reg = self.argReg(argnum)
        low_in = const & 0xffff
        high_in = (const >> 16) & 0xffff

        asm = "  movw " + arg_reg + ", #" + str(low_in) + "\n"
        asm = asm + "  movt " + arg_reg + ", #" + str(high_in) + "\n"
        return asm

