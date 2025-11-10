import sys
import logging

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import reg_tracker
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

class fngen:

    def __init__(self, gen : asmgen, rt : reg_tracker):
        self.gen = gen
        self.rt = rt

        self.required_loads : dict[str,int] = dict()

        self.debug_block : str = ""

        self.log = logging.getLogger("COMPOSER")

    def init_cc(self, cc : callconv, reverse_alias_map : dict[str,str] = dict()):
        params = cc.get_params()
        for name,(tag,idx) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if tag in ['greg','freg']:
                self.rt.reserve_specific_reg(tag, idx)
                self.rt.alias_reg(tag, name, idx)

        for name,(tag,idx) in params.items():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            if tag == 'sp':
                regidx = self.rt.reserve_any_reg('greg')
                self.rt.alias_reg('greg', name, regidx)
                self.required_loads[name] = idx

        self.debug_block = ""
        for name in params.keys():
            if name in reverse_alias_map:
                name = reverse_alias_map[name]
            reg = self.gen.greg(self.rt.aliased_regs['greg'][name])
            self.debug_block += self.gen.asmwrap(f"# {reg} = {name}")
            self.log.debug(f"Allocated {reg} for {name}")

    def get_boilerplate(self, cc : callconv):
        used_gregs = [v for _,v in self.rt.aliased_regs['greg'].items()]
        sr_count = len(set(used_gregs).intersection(cc.callee_save_lists['greg']))
        saveblock = cc.save_in_call(self.gen, regs={'greg':used_gregs})

        loadblock = ''

        for name,off in self.required_loads.items():
            loadblock += self.gen.asmwrap(f"# loading {name}")
            loadblock += self.gen.load_greg(
                    areg=self.gen.greg(cc.spreg),
                    offset=off+sr_count*8, # Assuming 64 bit/ 8 byte pointers
                    dst=self.gen.greg(self.rt.aliased_regs['greg'][name])
            )


        restoreblock = cc.restore_before_ret(self.gen, regs={'greg':used_gregs})

        return saveblock,self.debug_block+loadblock,restoreblock

def fngen_blis_gemm_ukr(
        init : str,
        fini : str,
        preload : str,
        main : str,
        nextpreload : str,
        store : str,
        rt : reg_tracker,
        gen):


    cc : callconv = gen.create_callconv()

    cc.add_param('greg', "m")
    cc.add_param('greg', "n")
    cc.add_param('greg', "k")
    cc.add_param('greg', "ADDR:alpha0")
    cc.add_param('greg', "ADDR:A0")
    cc.add_param('greg', "ADDR:B0")
    cc.add_param('greg', "ADDR:beta0")
    cc.add_param('greg', "ADDR:C0")
    cc.add_param('greg', "rs_c")
    cc.add_param('greg', "cs_c")
    cc.add_param('greg', "data")
    cc.add_param('greg', "cntx")

    params = cc.get_params()
    for name,(tag,idx) in params.items():
        if tag in ['greg','freg']:
            rt.reserve_specific_reg(tag, idx)
            rt.alias_reg(tag, name, idx)

    for name,(tag,idx) in params.items():
        if tag == 'sp':
            idx = rt.reserve_any_reg('greg')
            rt.alias_reg('greg', name, idx)

    for name in params.keys():
        reg = gen.greg(params[name][1])
        print(f"Allocated {reg} for {name}")

    sys.exit(-1)

    loopidx = rt.reserve_any_reg("greg")

    mainloop = loop(name="mainloop", counter=loopidx)

    if distance_pfc:
        asmblock = ""
        vecdim=0
        count = [m,n][vecdim]
        for i in m:
            asmblock += ""
        loop.add_singleshot_divergence(name="pfc", block="")
