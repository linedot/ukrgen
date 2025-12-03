# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from ukrgen.specializers.asm import op_support
from ukrgen.components.tile import tile

from ukrgen.models.load_store_operations import lsc_offset as lsco
from ukrgen.models.offset_mapper import offset_mapper

from asmgen.asmblocks.operations import widening_method as wm
from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import adt_size

def calculate_addr_parameters(sup : op_support,
                              primary_op : str,
                              gen : asmgen,
                              addr_reg_counts : dict[str,int],
                              data_tiles : dict[str, tile],
                              real_tiles : dict[str, tile],
                              narrow_components : list[str],
                              wide_components : list[str],
                              m : int, n : int, k : int,
                              vecdims : dict[str,int],
                              strides : dict[str,tuple[int,int]],
                              mappers : dict[str,offset_mapper],
                              strats : dict[str,str]):

    zo = lsco.zero_offset()
    components = list(addr_reg_counts.keys())

    ranges = dict()
    steps = dict()
    starts = dict()

    ways = adt_size(sup.triple.c)//adt_size(sup.triple.a)

    for c in components:
        data_t = data_tiles[c]
        real_t = real_tiles[c]
        dt = sup.triple.c
        if c in narrow_components:
            dt = sup.triple.a
        count = addr_reg_counts[c]

        max_val = gen.max_fload_immoff(dt=dt)
        if (real_t.is_tile or real_t.is_vector) and \
            data_t.is_scalar:
            max_val = gen.max_bcast_immoff(dt=dt)
        elif data_t.is_vla_tile or data_t.is_vla_vector:
            max_val = gen.max_load_voff
        elif data_t.is_fixed_tile or data_t.is_fixed_vector:
            max_val = gen.max_load_immoff(dt=dt)

        if getattr(gen,primary_op).widening_method == wm.SPLIT_INSTRUCTIONS \
           and c in wide_components:
            max_val //= ways

        if data_t.is_scalar or data_t.is_fixed_vector:
            ranges[c] = [(zo,lsco.so(max_val)) for i in range(count)]
        elif data_t.is_vla_vector or data_t.is_vla_tile:
            # TODO: differentiate VLEN/RLEN/etc...
            ranges[c] = [(zo,lsco.vo(0,max_val)) for i in range(count)]

        step = mappers[c].get_ldst_size(data_t)

        if "interleave" == strats[c]:
            steps[c] = [sum([step for jj in range(count)],zo) \
                    for j in range(count)]
            starts[c] = [sum([step for jj in range(j)],zo) \
                    for j in range(count)]

        elif "split" == strats[c]:
            full = [m,n,k][vecdims[c]]
            part = full//count
            splits = [part*j for j in range(count)]

            # TODO: handle steps/ranges > max_load_voff/immoff/etc...

            steps[c] = [sum([step for jj in range(full)],zo) \
                    for j in range(count)]
            starts[c] = [sum([step for jj in range(j)],zo) \
                    for j in splits]

            # Force split through range
            ranges[c] = [(zo, sum([step for jj in range(part)],zo)) \
                    for j in range(count)]
        elif "phase" == strats[c]:
            full = [m*n,n*m,k*m*n][vecdims[c]]

            steps[c] = [sum([step for jj in range(min(full,max_val+1))],zo) \
                    for j in range(count)]
            starts[c] = [zo for j in range(count)]
        else:
            raise NotImplementedError(f"multiaddr strategy \"{strat}\" not implemented")

    return starts,ranges,steps
