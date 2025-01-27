import unittest

from algobuild.load import (
        load_register_sequence_generator,
        lrsg_simple,
        )

from algobuild.compute.matmul import mm_mxn_register_sequence_generator

from asmgen.asmblocks.noarch import asm_data_type as adt
from asmgen.asmblocks.noarch import reg_tracker
from asmgen.asmblocks.noarch import asmgen

from typing import Type

class fma_mm:
    def __init__(self, dt : adt, gen : asmgen, 
                 abc_address_register_counts : list[int],
                 address_offset_splits : list[int],
                 abc_data_register_counts : list[int],
                 abc_preload_counts : list[int],
                 compute_rsg : Type[mm_mxn_register_sequence_generator],
                 abc_lrsg_types : list[Type[load_register_sequence_generator]],
                 m : int, n : int, unroll_factor : int,
                 use_fma_vf : bool,
                 max_outstanding_loads : int = 32,
                 ):

        self.m = m
        self.n = n
        self.unroll_factor = unroll_factor
        self.use_fma_vf = use_fma_vf
        self.max_outstanding_loads = max_outstanding_loads

        start_regs = [sum(abc_data_register_counts[:i]) for i in range(len(abc_data_register_counts))]
        if use_fma_vf:
            # That's a "too clever" oneliner, so lets do something that expresses the intent better
            # start_regs = [sum(abc_data_register_counts[:max(0,i-1)]) for i in range(len(abc_data_register_counts))]
            # B
            start_regs[1] = 0
            # C
            start_regs[2] = abc_data_register_counts[0]

        # TODO: rt.reserve_any_{v,f}reg ?? combined with use_fma_vf check? to replace the logic above?
        drls = [[i+start for i in range(count)]\
                for count,start in zip(abc_data_register_counts,start_regs)]

        parameters = {
            "a_data_register_list" : drls[0],
            "b_data_register_list" : drls[1],
            "c_data_register_list" : drls[2],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "c_data_register_list_start" : 0,
            "m" : m,
            "n" : n
        }
        side_effect_handlers = {
        }

        self.csg = compute_rsg(parameters=parameters, 
                          side_effect_handlers=side_effect_handlers)


        rt = reg_tracker(gen.max_gregs, gen.max_vregs, gen.max_fregs)


        arls = [[rt.reserve_any_greg() for i in range(count)]\
                for count in abc_address_register_counts]

        asmblock = ""

        # Default: vector offset
        # TODO: This needs to be handled better/combined with gen.isaquirks, etc...
        #       (This is still not the full generator, but just the mxm block generator,
        #       so isaquirks will be called before. Since I already forgot how vlenreg is
        #       stored: TODO: change how isaquirks works)
        offset_step = 1
        if 0 == gen.max_load_voff:
            # byte offset
            vlenreg = rt.reserve_any_greg()
            
            asmblock += gen.vsetvlmax(vlenreg, dt)

        self.lsgs = []
        thlds = [gen.max_load_voff, gen.max_fload_immoff(dt) if use_fma_vf else gen.max_load_voff, gen.max_load_voff]
        steps = [1, dt.value if use_fma_vf else 1, 1]
        for lsg_type,arl,drl,thld,count,step in \
                zip(abc_lrsg_types, arls, drls, thlds, abc_address_register_counts, steps):
            parameters = {
                "address_register_list" : arl,
                "address_register_list_start" : 0,
                "assumed_offset_start_list" : [i*step for i in range(count)],
                "offset_start_list" : [0 for i in range(count)],
                "offset_threshold_list" : [thld for i in range(count)],
                "offset_reset_value_list" : [0 for i in range(count)],
                "offset_step" : step,
                "max_offset" : thld,
                "data_register_list" : drl,
                "data_register_list_start" : 0
            }
            side_effect_handlers = {
            }

            self.lsgs.append(lsg_type(parameters=parameters, 
                             side_effect_handlers=side_effect_handlers))

        self.offset_factors = [1, dt.value if use_fma_vf else 1, 1]
        # current_data_origins, this is a list of dicts because a,b,c could use different register types
        self.cdos = [
            { r : i*ofac if i < precount else -1 for i,r in enumerate(drl)}\
                    for drl,precount,ofac in zip(drls,abc_preload_counts,self.offset_factors)
        ]

        # For loop version in case debugging is necessary
        # current_offsets = {}
        # for rlist,split in zip(arls, address_offset_splits):
        #     for i,r in enumerate(rlist):
        #         current_offsets[r] = i*split

        # TODO: When writing documentation, it needs to be communicated whether the splits 
        #       are always in # loads/vectors or directly in # bytes when fma_vf (or some
        #       possible other future method that would offset in bytes)
        self.current_offsets = { r : i*split*ofac for rlist,split,ofac in\
                zip(arls,address_offset_splits,self.offset_factors) for i,r in enumerate(rlist) }


    def generate(self):
        fma_sequence = [self.csg.advance() for i in range(self.m*self.n*self.unroll_factor)]

        outstanding_loads = []
        # TODO: Are multiple types of data resolvers necessary? Or would that be handled by a scheduler?
        # resolve just before it's needed
        def resolve_data(lrsg, outstanding_loads, dreg, toff,
                         current_offsets, current_data_origins,
                         dregsuff="d", off_factor = 1):
            for i in range(self.max_outstanding_loads):
                corg = current_data_origins[dreg]
                if corg == toff:
                    return
                for idx,target_state in enumerate(outstanding_loads):
                    tlareg = target_state["address_register"]
                    tldreg = target_state["data_register"]
                    tloff  = target_state["offset"]
                    taoff  = target_state["assumed_offset"]
                    #print(f"NEXT IN QUEUE: load of o{tloff+taoff} into {dregsuff}{tldreg} via a{tlareg}")
                    if (tldreg == dreg) and (tloff+taoff == toff):
                        caoff = current_offsets[tlareg]
                        if caoff != taoff:
                            print(f"SCHEDULE a{tlareg} <- a{tlareg}+o{taoff-caoff}")
                            current_offsets[tlareg] = taoff
                        if toff == tloff+taoff:
                            print(f"SCHEDULE {dregsuff}{dreg} <- LOAD a{tlareg}+o{tloff}")
                            current_data_origins[dreg] = taoff+tloff
                            del outstanding_loads[idx]
                            return
                outstanding_loads.append(lrsg.advance())
                #print("--------------------------------------------------")
            raise RuntimeError("Ran out of space for outstanding loads")

        
        print("==================================================")
        bpref = "v"
        if self.use_fma_vf:
            bpref = "f"
        for target_state in fma_sequence:
            regs = [target_state[reg] for reg in ["a_vector", "b_vector", "c_vector"]]
            target_offsets = [target_state[off] for off in ["a_data_offset", "b_data_offset", "c_data_offset"]]
            # Only a
            #regs = [target_state[reg] for reg in ["a_vector"]]
            #target_offsets = [target_state[off] for off in ["a_data_offset"]]
            for reg,toff,lsg,dos,suf,ofac in zip(regs,target_offsets,
                                                 self.lsgs,self.cdos,["v",bpref,"v"],
                                                 self.offset_factors):
                corg = dos[reg]
                if toff*ofac != corg:
                    #print(f"Data change o{corg} -> o{toff*ofac} for dreg {suf}{reg} required")
                    resolve_data(lsg, outstanding_loads, reg, toff*ofac, self.current_offsets, dos, suf, ofac)
            areg = regs[0]
            breg = regs[1]
            creg = regs[2]
            print(f"SCHEDULE v{creg} <- v{areg} x {bpref}{breg} + v{creg}")
        print("==================================================")


