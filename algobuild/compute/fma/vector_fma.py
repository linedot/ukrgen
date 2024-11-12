from asmgen.asmblocks.noarch import asmgen,asm_data_type

from .fma import fma, fma_info

from dataclasses import dataclass

from typing import Callable

class vector_scalar_fma(fma):
    def process(self):
        a = self.asmgen.vreg(self.fma_info.a_data_register)
        b = self.asmgen.freg(self.fma_info.b_data_register)
        c = self.asmgen.vreg(self.fma_info.c_data_register)
        dt = self.fma_info.datatype
        return self.asmgen.fma_vf(a, b, c, dt)

class vector_vector_lane_fma(fma):
    def __init__(self, asmgen, fma_info, lane_acquirer : Callable[[],int] = lambda : 0):
        self.lane_acquirer = lane_acquirer
        super().__init__(asmgen, fma_info)

    def process(self):
        a = self.asmgen.vreg(self.fma_info.a_data_register)
        b = self.asmgen.vreg(self.fma_info.b_data_register)
        c = self.asmgen.vreg(self.fma_info.c_data_register)
        dt = self.fma_info.datatype
        lane = self.lane_acquirer()
        return self.asmgen.fma_idx(a, b, c, lane, dt)

class vector_vector_fma(fma):
    def process(self):
        a = self.asmgen.vreg(self.fma_info.a_data_register)
        b = self.asmgen.vreg(self.fma_info.b_data_register)
        c = self.asmgen.vreg(self.fma_info.c_data_register)
        dt = self.fma_info.datatype
        return self.asmgen.fma(a, b, c, dt)
