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

from algobuild.components import *
from algobuild.generators import *
from algobuild.models import *


import traceback


class op_support:
    def __init__(self, triple : adt_triple,
                 a_tile : tile, b_tile : tile, c_tile : tile,
                 op_override : str = None):
        assert isinstance(triple, adt_triple)
        self.triple = triple
        self.a_tile = a_tile
        self.b_tile = b_tile
        self.c_tile = c_tile
        self.op_override = op_override
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

    def reset_analysis(self):
        self.vlenadds = set()
        self.byteadds = set()
        self.ops_used = []


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
            n_dp = dimension_properties(dt=dimension_type.fixed, size=ways,
                                        sdt=dimension_type.fixed, sd_size=ways)
            a_tile,b_tile,c_tile = tile(vec_dp, n_dp), tile(n_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
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
        if adt_size(dt_narrow) < adt_size(dt_wide):
            if arith_op.widening_method == wm.SPLIT_INSTRUCTIONS:
                additional_args['part'] = 0
                additional_args['modifiers'].add(mod.PART)
            elif arith_op.widening_method == wm.VEC_MULTI:
                additional_args['cdreg2'] = cdreg2
            # if we're dotting neighbours, we're actually doing mmas
            elif arith_op.widening_method == wm.DOT_NEIGHBOURS and\
                 op == 'fopa':
                op_to_append = 'mma'

        # Call to check if it raises an Exception
        arith_op(adreg=adreg, bdreg=adreg, cdreg=cdreg,
            a_dt=dt_narrow, b_dt=dt_narrow, c_dt=dt_wide,
            **additional_args)

        triple = adt_triple(a_dt=dt_narrow, b_dt=dt_narrow, c_dt=dt_wide) 
        at,bt,ct = self.op_tiles(op=op, triple=triple, arith_op=arith_op, modifiers=modifiers)

        self.op_support_map[op_to_append].append(
                op_support(triple=triple, a_tile=at, b_tile=bt, c_tile=ct))


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
            try:
                if op == 'fopa':
                    cdreg = self.gen.treg(0)
                    cdreg2 = self.gen.treg(1)
                elif op == 'mma':
                    adreg = self.gen.treg(0)
                    bdreg = self.gen.treg(1)
                    cdreg = self.gen.treg(2)
                    cdreg2 = self.gen.treg(3)
            except:
                continue
            arith_op = getattr(self.gen, op, None)
            if None == arith_op:
                continue
            fp_types = [dt for dt in adt if adt_is_float(dt)]
            int_types = [dt for dt in adt if adt_is_int(dt)]

            modifier_sets = [{mod.VF},set()]
            
            for type_list in [fp_types, int_types]:
                for dt_wide in type_list:
                    for dt_narrow in type_list:
                        for modifiers in modifier_sets:
                            if mod.VF in modifiers:
                                if self.gen.are_fregs_in_vregs:
                                    bdreg = self.gen.freg(3, dt_narrow)
                                else:
                                    bdreg = self.gen.freg(0, dt_narrow)
                            elif op != 'mma':
                                bdreg = bv
                            else:
                                bdreg = self.gen.treg(1)
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
            if isinstance(op, lsc_addr_add):
                if op.t.dima.dt == dimension_type.vla\
                    or op.t.dimb.dt == dimension_type.vla:
                    self.vlenadds.add(op.off)
                else:
                    self.byteadds.add(op.off)
            elif isinstance(op, lsc_transformation):
                self.ops_used.append(op.op)

    def pre_specialize(self, ops : list[lsc_operation], triple : adt_triple) -> list[lsc_operation]:
        # Changes needed:
        # - multiple instructions for widening lsc_transformation when wm = split_instructions
        #   - also change c idx for this case
        # - Don't change whem wm = vec_group. It isn't guaranteed that vector groups work
        #   like in RVV everywhere where you need multiple of ways as reg index
        ways = adt_size(triple.c)//adt_size(triple.a)
        if ways == 1:
            return ops
        if ways > 1:
            wms = [getattr(self.gen, op, None).widening_method for op in self.ops_used if getattr(self.gen, op, None) != None]
            if not any([wmtd==wm.SPLIT_INSTRUCTIONS for wmtd in wms]):
                return ops

        c_index = 2

        result_ops = []
        for op in ops:
            has_c = False
            for i,idx_list in enumerate(op.indices):
                if idx_list[0] == c_index and lsc_reg_type.data == op.reg_types[i]:
                    op.indices[i][1] *=ways
                    has_c = True
            if isinstance(op, lsc_addr_add):
                if op.rtype_idx == c_index:
                    op.off *= ways
            if has_c:
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
        return result_ops
