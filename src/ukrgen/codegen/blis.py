from asmgen.asmblocks.noarch import asmgen
from asmgen.callconv.callconv import callconv

def get_blis_gemm_cc(gen : asmgen):
    cc : callconv = gen.create_callconv()

    cc.add_param('greg', "m")
    cc.add_param('greg', "n")
    cc.add_param('greg', "k")
    cc.add_param('greg', "ADDR:alpha0")
    cc.add_param('greg', "ADDR:A0")
    cc.add_param('greg', "ADDR:B0")
    cc.add_param('greg', "ADDR:beta0")
    cc.add_param('greg', "ADDR:C0")
    cc.add_param('greg', "rs_C")
    cc.add_param('greg', "cs_C")
    cc.add_param('greg', "data")
    cc.add_param('greg', "cntx")

    return cc
