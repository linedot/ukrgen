# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import re
from typing import Self,Callable,Union

import traceback
import string

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import (
    reg_tracker,
    data_reg,
    asm_data_type as adt,
    adt_is_float,
    adt_is_int,
    adt_triple,
    adt_size,
)
from asmgen.asmblocks.operations import modifier as mod,widening_method as wm,opd3

from ..components import *
from ..generators import *
from ..models import *

from ..models.load_store_operations import lsc_reg_type



from .special_treg_ldst import special_treg_ldst,lsc_treg_row_store,lsc_treg_row_load

class op_support:
    def __init__(self, triple : adt_triple,
                 a_tile : tile, b_tile : tile, c_tile : tile,
                 target_op : str):
        assert isinstance(triple, adt_triple)
        self.triple = triple
        self.a_tile = a_tile
        self.b_tile = b_tile
        self.c_tile = c_tile
        self.target_op = target_op
    def __str__(self):
        v = lambda dim : "v" if dim.dt == dimension_type.vla else ""
        t = lambda dt : str(dt).replace("asm_data_type.","")
        triple = self.triple
        return (f"A: {t(triple.a)} with {self.a_tile.dima.size}{v(self.a_tile.dima)}x{self.a_tile.dimb.size}{v(self.a_tile.dimb)}\n"
                f"B: {t(triple.b)} with {self.b_tile.dima.size}{v(self.b_tile.dima)}x{self.b_tile.dimb.size}{v(self.b_tile.dimb)}\n"
                f"C: {t(triple.c)} with {self.c_tile.dima.size}{v(self.c_tile.dima)}x{self.c_tile.dimb.size}{v(self.c_tile.dimb)}\n")
    def __repr__(self):
        return self.__str__()

class lsc_specializer:
    def __init__(self, model : load_store_cpu, gen : asmgen, rt : reg_tracker):
        self.model = model
        self.gen = gen
        self.rt = rt

        self.reset_analysis()

        self.op_support_map = {}
        self.get_op_capabilities()

        self.transformations : dict[Type[lsc_operation],Callable[[lsc_operation,adt],str]] = {}

        self.transformations[lsc_addr_add] = self.transform_addr_add
        self.transformations[lsc_debugmsg] = lambda op,triple : op.msg
        self.transformations[lsc_load] = self.transform_load
        self.transformations[lsc_store] = self.transform_store
        self.transformations[lsc_transformation] = self.transform_transform
        self.transformations[lsc_zero] = self.transform_zero

        self.transformations[lsc_treg_row_load] = self.transform_trow_load
        self.transformations[lsc_treg_row_store] = self.transform_trow_store

    def determine_dreg_tag(self, dima : dimension_properties, dimb : dimension_properties) -> str:

        dreg_tag = 'vreg'

        if dima.dt == dimension_type.vla and \
           dimb.dt == dimension_type.vla:
            dreg_tag = 'treg'
        elif dima.dt == dimension_type.fixed and dima.size > 1 and \
             dimb.dt == dimension_type.fixed and dimb.size > 1:
            dreg_tag = 'treg'
        # TODO: This would for example be SME vector register when
        #       using widening instructions (dot neighbours + outer product)
        #       i.e. the vector register stores a matrix, but is not
        #       a tile register. Make decision and document cleanly
        #       how we want to handle this
        #elif dima.dt == dimension_type.vla and \
        #     dimb.dt == dimension_type.fixed and dimb.size > 1:
        #    dreg_tag = 'treg'
        #elif dima.dt == dimension_type.fixed and dima.size > 1 \
        #     dimb.dt == dimension_type.vla:
        #    dreg_tag = 'treg'
        elif dima.dt == dimension_type.fixed and dima.size == 1 and \
             dimb.dt == dimension_type.fixed and dimb.size == 1:
            dreg_tag = 'freg'
        else:
            dreg_tag = 'vreg'

        return dreg_tag

    def transform_transform(self, op : lsc_transformation, triple : adt_triple):

        dregs = []

        dreg_tags = []
        
        for i,(residx,subidx,t) in enumerate(zip(op.res_indices, op.sub_indices, op.tiles)):
            rtype_char = string.ascii_lowercase[i]

            dreg_tag = self.determine_dreg_tag(t.dima, t.dimb)

            residx = self.rt.aliased_regs[dreg_tag][f"{rtype_char}{residx}"]

            if dreg_tag in ["freg","treg"]:
                dreg = getattr(self.gen,dreg_tag)(residx,triple[i])
            else:
                dreg = getattr(self.gen,dreg_tag)(residx)

            dregs.append(dreg)
            dreg_tags.append(dreg_tag)

            # TODO: subidx

        more_args = dict()
        modifiers = set()

        opstr = op.op
        # If there is a number at the end of the str, that's a partial instruction
        numbers = re.findall(r'\d+',opstr)
        if numbers:
            modifiers.add(mod.PART)
            more_args['part'] = int(numbers[0])

            for number in numbers:
                opstr = opstr.replace(number, '')

        #if there is only 1 freg, this is a VF operation
        if 1 == dreg_tags.count('freg'):
            modifiers.add(mod.VF)
        
            


        arith_op = getattr(self.gen,opstr)

        return arith_op(adreg=dregs[0],bdreg=dregs[1],cdreg=dregs[2],
                        a_dt = triple.a, b_dt = triple.b, c_dt = triple.c,
                        modifiers=modifiers, **more_args)

        #return self.gen.asmwrap(f"dummy {op.op}")

    def transform_zero(self, op : lsc_zero, triple : adt_triple):

        
        rtype_char = string.ascii_lowercase[op.rtype_idx]

        dreg_tag = self.determine_dreg_tag(op.t.dima, op.t.dimb)

        residx = self.rt.aliased_regs[dreg_tag][f"{rtype_char}{op.res_idx}"]

        dt = triple[op.rtype_idx]

        if dreg_tag == 'freg':
            dreg = self.gen.freg(residx, dt)
            return self.gen.zero_freg(freg=dreg, dt=dt)
        elif dreg_tag == 'vreg':
            dreg = self.gen.vreg(residx)
            return self.gen.zero_vreg(vreg=dreg, dt=dt)
        elif dreg_tag == 'treg':
            dreg = self.gen.treg(residx, dt)
            return self.gen.zero_treg(treg=dreg, dt=dt)

        raise NotImplementedError(f"Implementation for zeroing {dreg_tag} missing")

    def transform_addr_add(self, op : lsc_operation, triple : adt_triple):

        rtype_char = string.ascii_lowercase[op.rtype_idx]
        aregidx = self.rt.aliased_regs['greg'][f"{rtype_char}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)
        data_t = op.tiles[1]
        dt = triple[op.rtype_idx]
        dt_bytes = adt_size(dt)
        if data_t.dima.dt == dimension_type.vla or\
           data_t.dimb.dt == dimension_type.vla:
            factor = op.off

            if factor < self.gen.max_add_voff:
                return self.gen.add_greg_voff(reg=areg, 
                                              offset=factor, 
                                              dt=triple[op.rtype_idx])

            vxnalias = "vlen"
            if op.off > 1:
                vxnalias = f"vlenx{factor}"
            vlen_idx = self.rt.aliased_regs['greg'][vxnalias]
            vlenreg = self.gen.greg(vlen_idx)
            return self.gen.add_greg_greg(dst=areg,
                                          reg1=areg,
                                          reg2=vlenreg)

        return self.gen.add_greg_imm(
            reg=areg,
            imm=op.off*dt_bytes)

    def transform_ldst(self, op : Union[lsc_load,lsc_store],
                       triple : adt_triple,
                       action : str):
        rtype_char = string.ascii_lowercase[op.rtype_idx]
        aregidx = self.rt.aliased_regs['greg'][f"{rtype_char}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        dt = triple[op.rtype_idx]
        dt_bytes = adt_size(dt)


        dreg_tag = self.determine_dreg_tag(op.t.dima, op.t.dimb)

        vlenmul = 0;
        if op.t.dima.dt == dimension_type.vla:
            vlenmul += 1
        if op.t.dimb.dt == dimension_type.vla:
            vlenmul += 1
            

        residx = self.rt.aliased_regs[dreg_tag][f"{rtype_char}{op.res_idx}"]

        if dreg_tag in ["freg","treg"]:
            dreg = getattr(self.gen,dreg_tag)(residx,dt)
        else:
            dreg = getattr(self.gen,dreg_tag)(residx)

        
        #TODO:
        # - differentiate voffset,immoffset, nonoffset, ...
        # - differentiate tile/vreg/freg

        if 0 == op.off:
            if 'treg' == dreg_tag:
                lsfunc = getattr(self.gen, f"{action}_tile")
                return lsfunc(areg=areg,treg=dreg,dt=dt)
            if 'vreg' == dreg_tag:
                lsfunc = getattr(self.gen, f"{action}_vector")
                return lsfunc(areg=areg,vreg=dreg,dt=dt)

        if 'freg' == dreg_tag:
            lsfunc = getattr(self.gen, f"{action}_scalar_immoff")
            return lsfunc(
                    areg=areg,
                    offset=op.off*dt_bytes,
                    freg=dreg,
                    dt=dt)

        if 'vreg' == dreg_tag:
            if vlenmul > 0:
                lsfunc = getattr(self.gen, f"{action}_vector_voff")
                return lsfunc(areg=areg, voffset=op.off, vreg=dreg, dt=dt)
            else:
                lsfunc = getattr(self.gen, f"{action}_vector_immoff")
                return lsfunc(areg=areg, offset=op.off, vreg=dreg, dt=dt)



        #raise NotImplementedError("Load not implemented yet")
        return self.gen.asmwrap(f"fixme: {action} {dreg_tag} with off {op.off} not implemented")

    def transform_load(self, op : lsc_load, triple : adt_triple):
        return self.transform_ldst(op=op,triple=triple,action="load")

    def transform_store(self, op : lsc_load, triple : adt_triple):
        return self.transform_ldst(op=op,triple=triple,action="store")

    def transform_trow_load(self, op : lsc_treg_row_load, triple : adt_triple):
        rtype_char = string.ascii_lowercase[op.rtype_idx]
        aregidx = self.rt.aliased_regs['greg'][f"{rtype_char}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        dt = triple[op.rtype_idx]

        asmblock = ""

        if 'rreg' not in self.rt.aliased_regs['greg']:
            # TODO: This needs to be 12-15 for SME. Figure out a way to 
            #       have generic register allocation restrictions
            # TODO: also potentially need multiple regs for different rtype_idx?
            rreg_id = 12
            self.rt.reserve_specific_reg('greg',rreg_id)
            self.rt.alias_reg('greg', 'rreg', rreg_id)
            rreg = self.gen.greg(rreg_id)
            asmblock += self.gen.mov_greg_imm(reg=rreg, imm=0)

        rreg_id = self.rt.aliased_regs['rreg']
        rreg = self.gen.greg(rreg_id)

        residx = self.rt.aliased_regs['treg'][f"{rtype_char}{op.res_idx}"]

        asmblock += self.gen.load_tile_row(
                       areg=areg, 
                       rreg=rreg,
                       roff=op.roff,
                       voff=op.aoff,
                       treg=self.gen.treg(residx),
                       dt=triple[op.rtype_idx])
        return asmblock

    def transform_trow_store(self, op : lsc_treg_row_store, triple : adt_triple):
        rtype_char = string.ascii_lowercase[op.rtype_idx]
        aregidx = self.rt.aliased_regs['greg'][f"{rtype_char}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        dt = triple[op.rtype_idx]

        asmblock = ""

        if 'rreg' not in self.rt.aliased_regs['greg']:
            # TODO: This needs to be 12-15 for SME. Figure out a way to 
            #       have generic register allocation restrictions
            rreg_id = 12
            self.rt.reserve_specific_reg('greg',rreg_id)
            self.rt.alias_reg('greg', 'rreg', rreg_id)
            rreg = self.gen.greg(rreg_id)
            asmblock += self.gen.mov_greg_imm(reg=rreg, imm=0)

        rreg_id = self.rt.aliased_regs['greg']['rreg']
        rreg = self.gen.greg(rreg_id)

        residx = self.rt.aliased_regs['treg'][f"{rtype_char}{op.res_idx}"]

        asmblock += self.gen.store_tile_row(
                       areg=areg, 
                       rreg=rreg,
                       roff=op.roff,
                       voff=op.aoff,
                       treg=self.gen.treg(residx, dt=dt),
                       dt=dt)
        return asmblock

    def reset_analysis(self):
        self.vlenadds = set()
        self.byteadds = set()
        self.ops_used = []
        self.address_registers : dict[int,set[int]] = dict()
        self.data_registers : dict[int,set[int]] = dict()
        self.data_tags : dict[int,str] = dict()


    def op_tiles(self, op : str, triple : adt_triple, arith_op : opd3, modifiers : set[mod]):
        # c/a
        ways = adt_size(triple.c)//adt_size(triple.a)
        vec_dt = dimension_type.fixed
        if self.gen.is_vla:
            vec_dt = dimension_type.vla
            vec_size = 1
        else:
            vec_size = self.gen.simd_size//adt_size(triple.c)

        # TODO: indexable elements (like 128bit in SVE)
        vec_dp = dimension_properties(dt=vec_dt, size=vec_size,
                                      sdt=vec_dt, sd_size=vec_size)
    
        wide_vec_dp = vec_dp
        # TODO: not sure how to better handle it - the mm generator
        #       right now needs the sizes to be the same, but we will
        #       need waysx registers for C
        #if arith_op.widening_method in [wm.VEC_GROUP, wm.VEC_MULTI]:
        #    wide_vec_dp = dimension_properties(dt=vec_dt, size=vec_size*ways,
        #                                       sdt=vec_dt, sd_size=vec_size*ways)

        # Add the tiles for this triple
        if arith_op.widening_method == wm.DOT_NEIGHBOURS and\
          op == 'fopa' and\
          ways > 1:
            # This makes things more complicated for offset computation
            #n_dp = dimension_properties(dt=dimension_type.fixed, size=ways,
            #                            sdt=dimension_type.fixed, sd_size=ways)
            #a_tile,b_tile,c_tile = tile(vec_dp, n_dp), tile(n_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
            a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(scalar_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
        elif op == 'fopa':
            a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(scalar_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
        elif op == 'fma' or op == 'fmul':
            a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(vec_dp, scalar_dp), tile(wide_vec_dp, scalar_dp)
        elif op == 'dota':
            a_tile,b_tile,c_tile = tile(scalar_dp, vec_dp), tile(vec_dp, scalar_dp), tile(scalar_dp, scalar_dp)
        else:
            raise NotImplementedError(f"Adding tiles for {op} not implemented yet")

        if mod.VF in modifiers:
            b_tile = tile(scalar_dp, scalar_dp)

        return a_tile,b_tile,c_tile

    def add_if_supported(self, op : str, arith_op : opd3, dt_narrow : adt, dt_wide : adt,
                         adreg : data_reg, bdreg : data_reg, cdreg : data_reg, cdreg2 : data_reg,
                         modifiers : set[mod]):
        op_to_append = op
        additional_args = {}
        additional_args['modifiers'] = modifiers

        target_op = op
        if adt_size(dt_narrow) < adt_size(dt_wide):
            if arith_op.widening_method == wm.SPLIT_INSTRUCTIONS:
                additional_args['part'] = 0
                additional_args['modifiers'].add(mod.PART)
            elif arith_op.widening_method == wm.VEC_MULTI:
                additional_args['cdreg2'] = cdreg2
            # if we're dotting neighbours, we're actually doing mmas
            elif arith_op.widening_method == wm.DOT_NEIGHBOURS and\
                 op == 'fopa':
                target_op = 'mma'

        # Call to check if it raises an Exception
        arith_op(adreg=adreg, bdreg=adreg, cdreg=cdreg,
            a_dt=dt_narrow, b_dt=dt_narrow, c_dt=dt_wide,
            **additional_args)

        triple = adt_triple(a_dt=dt_narrow, b_dt=dt_narrow, c_dt=dt_wide) 
        at,bt,ct = self.op_tiles(op=op, triple=triple, arith_op=arith_op, modifiers=modifiers)

        self.op_support_map[op_to_append].append(
                op_support(triple=triple, a_tile=at, b_tile=bt, c_tile=ct, target_op=target_op))


    def get_op_capabilities(self):
        av = self.gen.vreg(0)
        bv = self.gen.vreg(1)
        cv = self.gen.vreg(2)
        cv2 = self.gen.vreg(3)

        ops = ['fma','fmul','fopa','dota','mma']
        for op in ops:
            self.op_support_map[op] = []
        for op in ops:
            # TODO: better way to allocate for the test
            adreg = av
            bdreg = bv
            cdreg = cv
            cdreg2 = cv2
            arith_op = getattr(self.gen, op, None)
            if None == arith_op:
                continue
            fp_types = [dt for dt in adt if adt_is_float(dt)]
            int_types = [dt for dt in adt if adt_is_int(dt)]

            modifier_sets = [{mod.VF},set()]
            
            for type_list in [fp_types, int_types]:
                for dt_wide in type_list:
                    for dt_narrow in type_list:
                        try:
                            if op == 'fopa':
                                cdreg = self.gen.treg(0, dt_wide)
                                cdreg2 = self.gen.treg(1, dt_wide)
                            elif op == 'mma':
                                adreg = self.gen.treg(0, dt_narrow)
                                bdreg = self.gen.treg(1, dt_narrow)
                                cdreg = self.gen.treg(2, dt_wide)
                                cdreg2 = self.gen.treg(3, dt_wide)
                        except:
                            continue
                        for modifiers in modifier_sets:
                            if mod.VF in modifiers:
                                if self.gen.are_fregs_in_vregs:
                                    bdreg = self.gen.freg(3, dt_narrow)
                                else:
                                    bdreg = self.gen.freg(0, dt_narrow)
                            elif op != 'mma':
                                bdreg = bv
                            else:
                                bdreg = self.gen.treg(1, dt_narrow)
                            if adt_size(dt_narrow) > adt_size(dt_wide):
                                continue
                            try:
                                self.add_if_supported(op=op, arith_op=arith_op,
                                                      dt_narrow=dt_narrow, dt_wide=dt_wide,
                                                      adreg=adreg, bdreg=bdreg, cdreg=cdreg, cdreg2=cdreg2,
                                                      modifiers=modifiers)
                            except Exception as e:
                                #print(e)
                                #print(traceback.format_exc())
                                pass

    def analyse(self, ops : list[lsc_operation]):
        for op in ops:
            if isinstance(op, lsc_addr_add) or \
               isinstance(op, lsc_load) or \
               isinstance(op, lsc_store):
                # reserve registers
                if op.rtype_idx not in self.address_registers:
                    self.address_registers[op.rtype_idx] = set()

                self.address_registers[op.rtype_idx].add(op.addr_idx)

            if isinstance(op, lsc_load) or \
               isinstance(op, lsc_store):
                # reserve registers
                if op.rtype_idx not in self.data_registers:
                    self.data_registers[op.rtype_idx] = set()

                dreg_tag = self.determine_dreg_tag(op.t.dima, op.t.dimb)

                self.data_registers[op.rtype_idx].add(op.res_idx)
                self.data_tags[op.rtype_idx] = dreg_tag

            if isinstance(op, lsc_addr_add):

                # vlen/byte adds
                if op.t.dima.dt == dimension_type.vla\
                    or op.t.dimb.dt == dimension_type.vla:
                    if op.off not in self.vlenadds:
                        if 1 == op.off:
                            continue
                        if self.gen.max_add_voff > op.off:
                            continue
                        self.vlenadds.add(op.off)
                        print(f"aliasing vlenx{op.off}")
                        regidx = self.rt.reserve_any_reg('greg')
                        self.rt.alias_reg('greg',
                                          f"vlenx{op.off}",
                                          regidx)
                else:
                    self.byteadds.add(op.off)
            elif isinstance(op, lsc_transformation):
                self.ops_used.append(op.op)
                for i,(residx,t) in enumerate(zip(op.res_indices,op.tiles)):

                    dreg_tag = self.determine_dreg_tag(t.dima, t.dimb)

                    if i not in self.data_tags:
                        self.data_tags[i] = dreg_tag
                    else:
                        if self.data_tags[i] != dreg_tag:
                            rtype_char = string.ascii_lowercase[i]
                            raise ValueError(f"{op.op} with {rtype_char}{residx} is a {dreg_tag}, but is already registered as {self.data_tags[i]}")

                    if i not in self.data_registers:
                        self.data_registers[i] = set()
                    self.data_registers[i].add(residx)


    def pre_specialize(self, ops : list[lsc_operation], triple : adt_triple) -> list[lsc_operation]:
        # Changes needed:
        # - multiple instructions for widening lsc_transformation when wm = split_instructions
        #   - also change c idx for this case
        # - Don't change whem wm = vec_group. It isn't guaranteed that vector groups work
        #   like in RVV everywhere where you need multiple of ways as reg index
        ways = adt_size(triple.c)//adt_size(triple.a)
        split_instructions = False
        if ways > 1:
            wms = [getattr(self.gen, op, None).widening_method for op in self.ops_used if getattr(self.gen, op, None) != None]
            if any([wmtd==wm.SPLIT_INSTRUCTIONS for wmtd in wms]):
                split_instructions = True

        c_index = 2


        stls = special_treg_ldst()
        need_stls = False

        result_ops = []
        for op in ops:
            if need_stls:
                if stls.check_load_flush(op):
                    result_ops.extend(stls.flush_load())
                    print(f"flushing treg loads")
                if stls.check_store_flush(op):
                    result_ops.extend(stls.flush_store())
                    print(f"flushing treg store")
            if isinstance(op, (lsc_load, lsc_store)):
                if (op.t.dima.size > 1 and op.t.dimb.size > 1) or \
                   (op.t.dima.dt == dimension_type.vla and \
                    op.t.dimb.dt == dimension_type.vla):
                    action = "load"
                    if isinstance(op, lsc_store):
                        action = "store"
                    try:
                        lsfunc = getattr(self.gen,f"{action}_tile")
                        lsfunc(areg=self.gen.greg(0),
                               treg=self.gen.treg(0),
                               dt=triple[op.rtype_idx])
                        result_ops.append(op)
                    except:
                        print(f"adding treg {action}")
                        need_stls = True
                        addfunc = getattr(stls, f"add_treg_{action}")
                        addfunc(op, triple[op.rtype_idx])

                    continue
            has_c = False
            if split_instructions:
                for i,idx_list in enumerate(op.indices):
                    if idx_list[0] == c_index and lsc_reg_type.data == op.reg_types[i]:
                        op.indices[i][1] *=ways
                        self.data_registers[c_index].add(op.indices[i][1])
                        for wayreg in range(1,ways):
                            self.data_registers[c_index].add(op.indices[i][1]+wayreg)
                        has_c = True
                if isinstance(op, lsc_addr_add):
                    if op.rtype_idx == c_index:
                        op.off *= ways
            if has_c and split_instructions:
                if need_stls:
                    raise NotImplementedError("split instruction and special tile ld/st not implemented")
                for i in range(ways):
                    part_op = copy.deepcopy(op)
                    for j,idx_list in enumerate(part_op.indices):
                        if idx_list[0] == c_index and lsc_reg_type.data == part_op.reg_types[j]:
                            part_op.indices[j][1] += i
                    if isinstance(op, lsc_transformation):
                        part_op.op += f"{i}"
                    if isinstance(op, lsc_load) or isinstance(op, lsc_store):
                        part_op.off = part_op.off*2+i
                    result_ops.append(part_op)
            else:
                result_ops.append(op)
        if need_stls:
            result_ops.extend(stls.flush_load())
            result_ops.extend(stls.flush_store())
        return result_ops

    def specialize(self, ops : list[lsc_operation], triple : adt_triple) -> list[str]:
        result = []
        for op in ops:
            transform = self.transformations[type(op)]
            asm = transform(op, triple)
            result.append(asm)
        return result

    def init_vlenregs(self, triple: adt_triple) -> str:

        asmblock = ""

        if not self.vlenadds:
            return ""
        if 'vlen' not in self.rt.aliased_regs['greg']:
            vlenidx = self.rt.reserve_any_reg('greg')
            self.rt.alias_reg('greg', 'vlen', vlenidx)
            vlenreg = self.gen.greg(vlenidx)
            asmblock += self.gen.simd_size_to_greg(reg=vlenreg, dt=triple.a)
        else:
            vlenidx = self.rt.aliased_regs['greg']['vlen']
            vlenreg = self.gen.greg(vlenidx)

        for vlenxn in self.vlenadds:

            vlenxnidx = self.rt.reserve_any_reg('greg')
            self.rt.alias_reg('greg', f"vlenx{vlenxn}", vlenxnidx)

            vlenxnreg = self.gen.greg(vlenxnidx)

            asmblock += self.gen.mul_greg_imm(src=vlenreg,
                                              dst=vlenxnreg,
                                              factor=vlenxn)

        return asmblock


    def code_init(self, triple : adt_triple) -> str:
        #TODO: datatype for quirks. For now go with the narrow type
        asmblock = self.gen.isaquirks(rt=self.rt,dt=triple.a)

        asmblock += self.init_vlenregs(triple=triple)

        # reserve address registers
        for rtype_idx in self.address_registers.keys():
            rtype_char = string.ascii_lowercase[rtype_idx]
            for aidx in self.address_registers[rtype_idx]:
                reg_alias = f"{rtype_char}areg{aidx}"
                aregidx = self.rt.reserve_any_reg('greg')
                self.rt.alias_reg('greg', reg_alias, aregidx)

                # asmblock += f"// todo: init {reg_alias}\n"
                asmblock += self.gen.asmwrap(f"; {self.gen.greg(aregidx)} = {reg_alias}")

        # reserve data registers
        for rtype_idx in self.data_registers.keys():
            rtype_char = string.ascii_lowercase[rtype_idx]
            for aidx in self.data_registers[rtype_idx]:
                reg_alias = f"{rtype_char}{aidx}"
                dreg_tag = self.data_tags[rtype_idx]
                dregidx = self.rt.reserve_any_reg(dreg_tag)
                self.rt.alias_reg(dreg_tag, reg_alias, dregidx)

                dt = triple.__dict__[string.ascii_lowercase[rtype_idx]]

                if dreg_tag in ["freg","treg"]:
                    dregname = str(getattr(self.gen,dreg_tag)(dregidx,dt))
                else:
                    dregname = str(getattr(self.gen,dreg_tag)(dregidx))

                # asmblock += f"// todo: init {reg_alias}\n"
                asmblock += self.gen.asmwrap(f"; {dregname} = {reg_alias}")

        return asmblock

    def code_fini(self, triple : adt_triple) -> str:
        return self.gen.isaendquirks(rt=self.rt,dt=triple.a)
