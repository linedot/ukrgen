# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import argparse
import sys
import logging

from mako.template import Template

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.asmblocks.operations import widening_method as wm


from asmgen.registers import (
        asm_data_type as adt,
        adt_triple,
        adt_size,
        reg_tracker
        )
from ukrgen.specializers.asm import lsc_specializer
from ukrgen.components import tile,simple_ukr_tile,dimension_type,dimension_properties
from ukrgen.components.tile import scalar_dp
from ukrgen.generators.mm import mm,order2D
from ukrgen.models.load_store_cpu import load_store_cpu
from ukrgen.models.load_store_operations import lsc_offset,stridexvlen,lsc_load,lsc_transformation,ldst_modifier
from ukrgen.models.addr_resolver import addr_resolver
from ukrgen.models.offset_mapper import flat_mapper,strided_mapper
from ukrgen.schedulers import simple_dependency_scheduler


asmgen_map = {
    'avx128' : fma128,
    'avx256' : fma256,
    'avx512' : avx512,
    'rvv' : rvv,
    'rvv071' : rvv071,
    'neon' : neon,
    'sve' : sve,
    'sme' : sme,
}

helpargs=['-h','--help']

def helpexit_if_last_parser(rest : list[str], parser : argparse.ArgumentParser):
    if any(a in rest for a in helpargs):
        if not [a for a in rest if a not in helpargs]:
            parser.print_help()
            sys.exit(0)


def parse_main_arguments(parser : argparse.ArgumentParser):

    operations = ['fma','dota','fopa','mma']

    kernels = ['mm']
    parser.add_argument("--ukr", type=str,
                        choices=kernels,
                        required=True, help="Kernel to generate")
    parser.add_argument("--isa", type=str,
                        choices=asmgen_map.keys(),
                        required=True, help="ISA to use")
    parser.add_argument("--op", type=str,
                        choices=operations,
                        required=True, help="Arithmetic operation")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest


def parse_specializer_arguments(
        specializer : lsc_specializer,
        op : str,
        parser : argparse.ArgumentParser):

    narrow_choices = []
    for sup in specializer.op_support_map[op]:
        narrow_choices.append(str(sup.triple.a).replace("asm_data_type.",""))

    parser.add_argument("--ab-data-type", type=str, required=True,
                        choices=narrow_choices, help="Type to use for A/B tiles")

    wide_choices = []
    for sup in specializer.op_support_map[op]:
        wide_choices.append(str(sup.triple.c).replace("asm_data_type.",""))

    parser.add_argument("--c-data-type", type=str, required=True,
                        choices=wide_choices, help="Type to use for A/B tiles")


    spec_args, rest = parser.parse_known_args()

    

    ab_type = adt[spec_args.ab_data_type]
    c_type = adt[spec_args.c_data_type]

    op_support_list = [sup for sup in specializer.op_support_map[op] if \
            (ab_type == sup.triple.a and \
             ab_type == sup.triple.b and \
             c_type == sup.triple.c)
            ]

    variantstr = "\n".join([f"{i}: <{str(sup)}>" for \
            i,sup in enumerate(op_support_list)])

    parser.add_argument("--variant", type=int, required=True,
                        choices=list(range(len(op_support_list))),
                        help=f"Variant to use. Details: {variantstr}")

    spec_args, rest = parser.parse_known_args()

    helpexit_if_last_parser(rest=rest, parser=parser)
    return spec_args, rest, op_support_list[spec_args.variant]

def parse_fma_args(parser : argparse.ArgumentParser):


    parser.add_argument("--fma-unvec-method", type=str, required=True,
                        choices=['vf','lane_select','lane_bcast','load_bcast'],
                        help="""
                        When using fma as base instruction and both A and B are vectors,
                        determines how scalars are accessed in the unvectorized dimension
                        """)

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_ukr_args(parser : argparse.ArgumentParser):


    parser.add_argument("--m", type=int, required=True,
                        help="m dimension in tiles")
    parser.add_argument("--n", type=int, required=True,
                        help="n dimension in tiles")
    parser.add_argument("--k", type=int,
                        default=1,
                        help="k dimension in tiles")
    parser.add_argument("--order", type=str,
                        default="mnkMNK",
                        help="Loop order (default=mnkMNK)")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_stride_args(parser : argparse.ArgumentParser):


    parser.add_argument("--column-strides", type=str, nargs="+", required=False,
                        choices=['a','b','c'],
                        help="components with general column strides")
    parser.add_argument("--row-strides", type=str, nargs="+", required=False,
                        choices=['a','b','c'],
                        help="components with general row strides")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_lsc_args(parser : argparse.ArgumentParser):


    multiareg_strategies = ["interleave","split"]

    parser.add_argument("--a-data-regs", type=int, required=True,
                        help="number of data registers to use for the a tiles")
    parser.add_argument("--b-data-regs", type=int, required=True,
                        help="number of data registers to use for the b tiles")
    parser.add_argument("--c-data-regs", type=int, required=True,
                        help="number of data registers to use for the c tiles")
    parser.add_argument("--a-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the a tiles")
    parser.add_argument("--a-multiaddr-strat", type=str,
                        default=multiareg_strategies[0],
                        choices=multiareg_strategies,
                        help="Strategy for using multiple address registers for a tiles")
    parser.add_argument("--b-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the b tiles")
    parser.add_argument("--b-multiaddr-strat", type=str,
                        default=multiareg_strategies[0],
                        choices=multiareg_strategies,
                        help="Strategy for using multiple address registers for b tiles")
    parser.add_argument("--c-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the c tiles")
    parser.add_argument("--c-multiaddr-strat", type=str,
                        default=multiareg_strategies[0],
                        choices=multiareg_strategies,
                        help="Strategy for using multiple address registers for c tiles")
    parser.add_argument("--a-preload", type=int, required=True,
                        help="number of a data registers to preload")
    parser.add_argument("--b-preload", type=int, required=True,
                        help="number of b data registers to preload")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_sched_args(parser : argparse.ArgumentParser):

    parser.add_argument("--sched-rar-distance", type=int, default=0,
                        help="Minimum read-after-read distance")
    parser.add_argument("--sched-raw-distance", type=int, default=0,
                        help="Minimum read-after-write distance")
    parser.add_argument("--sched-war-distance", type=int, default=0,
                        help="Minimum write-after-read distance")
    parser.add_argument("--sched-waw-distance", type=int, default=0,
                        help="Minimum write-after-waw distance")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_prefetch_args(parser : argparse.ArgumentParser):

    pf_c_strats = ["none","pre_loop", "post_loop", "distance"]
    parser.add_argument("--pf-c-strat", type=str, choices=pf_c_strats,
                        default="none",
                        help="Strategy for prefetching C component into memory")
    parser.add_argument("--pf-c-distance", type=int, required=False,
                        help="distance in loop iterations when c is prefetched before the loop finishes (strat must be \"distance\")")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_inout_args(parser : argparse.ArgumentParser):
    parser.add_argument("--output-filename", type=str, required=True,
                        help="Path to the file to output the result to")
    parser.add_argument("--input-filename", type=str, required=True,
                        help="""Path to the sourcefile containing the template to
                        fill with kernel and parameters
                        """)

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def main():

    logging.basicConfig(level=logging.DEBUG)

    stridelog = logging.getLogger("STRIDE")
    stridelog.setLevel(logging.DEBUG)

    tiflog = logging.getLogger("TIF")
    tiflog.setLevel(logging.DEBUG)

    asmlog = logging.getLogger("ASM")
    asmlog.setLevel(logging.DEBUG)

    lsclog = logging.getLogger("LSC")
    lsclog.setLevel(logging.DEBUG)

    genlog = logging.getLogger("GENERATOR")
    genlog.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
            description="ukrgen compute kernel generator",
            add_help=False)

    args,rest = parse_main_arguments(parser=parser)

    gen = asmgen_map[args.isa]()

    rt = reg_tracker([('greg',gen.max_gregs),
                      ('freg',gen.max_fregs),
                      ('vreg',gen.max_vregs),
                      ('treg',gen.max_tregs(adt.FP64))])

    specializer = lsc_specializer(model=None, gen=gen, rt=rt)

    #print("Generator supports following operations:")
    #for k,v in specializer.op_support_map.items():
    #    print(f"Operation {k}:")
    #    for sup in v:
    #        print(f"{sup}")
    #print(f"Operation support for {args.op}:")
    #for sup in specializer.op_support_map[args.op]:
    #    print(f"{sup}")


    spec_args,rest,sup = parse_specializer_arguments(
            specializer=specializer, op=args.op, parser=parser)

    genlog.debug("Chosen variant:")
    genlog.debug(f"{sup}")

    triple = adt_triple(a_dt=adt[spec_args.ab_data_type],
                        b_dt=adt[spec_args.ab_data_type],
                        c_dt=adt[spec_args.c_data_type])

    ways = adt_size(triple.c)//adt_size(triple.a)


    ukr_args,rest = parse_ukr_args(parser=parser)

    m,n,k = ukr_args.m,ukr_args.n,ukr_args.k
    order = order2D(ukr_args.order)

    nc=n
    nb=n
    fma_args = None
    if args.op == 'fma' and sup.b_tile.dima == sup.a_tile.dima:
        fma_args, rest = parse_fma_args(parser=parser)
        if fma_args.fma_unvec_method in ['vf','load_bcast']:
            sup.b_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
        elif fma_args.fma_unvec_method in ['lane_select','lane_bcast']:

            raise NotImplementedError("lane_select and lane_bcast not yet implemented")
            b_into_size = gen.indexable_elements(sup.triple.b)
            nb //= b_into_size
            sup.b_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
            sup.b_tile.dimb = dimension_properties(
                    dt=dimension_type.fixed, size=b_into_size,
                    sdt=dimension_type.fixed, sd_size=b_into_size)


    a_tile = simple_ukr_tile(a_size=m, b_size=k,
                             subdims=(sup.a_tile.dima, sup.a_tile.dimb))
    b_tile = simple_ukr_tile(a_size=k, b_size=nb,
                             subdims=(sup.b_tile.dima, sup.b_tile.dimb))
    c_tile = simple_ukr_tile(a_size=m, b_size=nc,
                             subdims=(sup.c_tile.dima, sup.c_tile.dimb))

    genmm = mm(a=a_tile, b=b_tile, c=c_tile, lo=order, opstr=args.op)

    scale_tile = simple_ukr_tile(a_size=m*n, b_size=1,
                                 subdims=(sup.c_tile.dima,
                                          sup.c_tile.dimb))
    alphabeta_tile = simple_ukr_tile(a_size=1, b_size=1,
                                subdims=(scalar_dp,scalar_dp))
    genbetascale = mm(scale_tile, alphabeta_tile, scale_tile,
                  opstr="fmulnp", tile_strs=["C","beta","C"])
    genalphascale = mm(scale_tile, alphabeta_tile, scale_tile,
                    opstr="fma", tile_strs=["AB","alpha","C"])

    mm_ops = genmm.generate()

    tiflog.debug("################### TILE-INSTRUCTION-FORMAT ###################")
    for op in mm_ops:
        tiflog.debug(str(op))

    stride_args,rest = parse_stride_args(parser=parser)

    lsc_args,rest = parse_lsc_args(parser=parser)


    addr_indices = [[i for i in range(count)]
                    for count in [lsc_args.a_addr_regs,
                                  lsc_args.b_addr_regs,
                                  lsc_args.c_addr_regs]]


    is_tile_scalar = lambda t : t.dima.dt == dimension_type.fixed and \
                                t.dima.size == 1 and \
                                t.dimb.dt == dimension_type.fixed and \
                                t.dimb.size == 1

    is_tile_vla_vector = lambda t : (t.dima.dt == dimension_type.vla and \
                                     t.dima.size == 1 and \
                                     t.dimb.dt == dimension_type.fixed and \
                                     t.dimb.size == 1) or \
                                    (t.dima.dt == dimension_type.fixed and \
                                     t.dima.size == 1 and \
                                     t.dimb.dt == dimension_type.vla and \
                                     t.dimb.size == 1)

    is_tile_vla_tile = lambda t : (t.dima.dt == dimension_type.vla and \
                                   t.dima.size == 1 and \
                                   t.dimb.dt == dimension_type.vla and \
                                   t.dimb.size == 1)

    zo = lsc_offset.zero_offset()
    addr_offset_ranges=[]
    addr_offset_steps=[]
    addr_offset_starts=[]
    for i,(t,dt,count) in enumerate(zip(
        [sup.a_tile,
         sup.b_tile,
         sup.c_tile],
        [sup.triple.a,
         sup.triple.b,
         sup.triple.c],
        [lsc_args.a_addr_regs,
         lsc_args.b_addr_regs,
         lsc_args.c_addr_regs]
        )):
        if is_tile_vla_tile(t):
            vmax=gen.max_load_voff
            addr_offset_ranges.append([(zo, lsc_offset({},[],[0,vmax],0)) for j in range(count)])
            addr_offset_steps.append([lsc_offset({},[],[0,(vmax+1)*count],0) for j in range(count)])
            addr_offset_starts.append([lsc_offset({},[],[0,j*t.dima.size],0) for j in addr_indices[i]])
        elif is_tile_vla_vector(t):
            vmax=gen.max_load_voff
            addr_offset_ranges.append([(zo, lsc_offset({},[],[vmax],0)) for j in range(count)])
            addr_offset_steps.append([lsc_offset({},[],[(vmax+1)*count],0) for j in range(count)])
            addr_offset_starts.append([lsc_offset({},[],[j*t.dima.size],0) for j in addr_indices[i]])
        elif is_tile_scalar(t):
            imax=gen.max_fload_immoff(dt=dt)
            addr_offset_ranges.append([(zo, lsc_offset({},[],[],imax)) for j in range(count)])
            addr_offset_steps.append([lsc_offset({},[],[],(imax+1)*count) for j in range(count)])
            addr_offset_starts.append([lsc_offset({},[],[],j*t.dima.size) for j in addr_indices[i]])


    # Ensure the specializer doesn't generate impossible voffsets for loads/stores of
    # C regs
    if getattr(gen, args.op).widening_method == wm.SPLIT_INSTRUCTIONS:
        for i in range(len(addr_offset_ranges[2])):
            maxoff = addr_offset_ranges[2][i][1]
            maxoff.vlen_strides = [v//ways for v in maxoff.vlen_strides]
            maxoff.reg_strides = [r//ways for r in maxoff.reg_strides]
            maxoff.immoff //= ways
            addr_offset_ranges[2][i] = (addr_offset_ranges[2][i][0],
                                        maxoff)

        #for i,_ in enumerate(addr_offset_steps[2]):
        #    addr_offset_steps[2][i].vlen_strides = \
        #        [v//ways for v in addr_offset_steps[2][i].vlen_strides]
        #    addr_offset_steps[2][i].reg_strides = \
        #        [r//ways for r in addr_offset_steps[2][i].reg_strides]
        #    addr_offset_steps[2][i].immoff //= ways

    strides = {k : (None,None) for k in ['a','b','c']}
    i = 0
    for char in ['a','b','c']:
        comp_slist = [None,None]
        if stride_args.row_strides:
            if char in stride_args.row_strides:
                stridelog.debug(f"Component {char} has row stride")
                comp_slist[0] = i
                i+= 1
        if stride_args.column_strides:
            if char in stride_args.column_strides:
                stridelog.debug(f"Component {char} has col stride")
                comp_slist[1] = i
                i+= 1

        strides[char] = tuple(comp_slist)


    a_mapper = strided_mapper((m,1), strides['a'],vecdim=0)
    b_mapper = strided_mapper((1,n), strides['b'],vecdim=1)
    c_mapper = strided_mapper((m,n), strides['c'],vecdim=0)

    # TODO: arbitrary vectorization direction
    # TODO: Logic might be almost the same for no strides, possibly making the block further above redundant
    for i,(char,vecdim,mapper,t,count,strat) in enumerate(zip(
            ['a','b','c'],
            [0,1,0],
            [a_mapper,b_mapper,c_mapper],
            [sup.a_tile,sup.b_tile,sup.c_tile],
            [lsc_args.a_addr_regs,
             lsc_args.b_addr_regs,
             lsc_args.c_addr_regs],
            [lsc_args.a_multiaddr_strat,
             lsc_args.b_multiaddr_strat,
             lsc_args.c_multiaddr_strat]
            )):
        step = mapper.get_ldst_size(t)
        if "interleave" == strat:
            print(f"Interleave strat for {char}")
            addr_offset_steps[i] = [sum([step for jj in range(count)],lsc_offset.zero_offset()) \
                    for j in range(count)]
            addr_offset_starts[i] = [sum([step for jj in range(j)],lsc_offset.zero_offset()) \
                    for j in range(count)]
        elif "split" == strat:
            print(f"Split strat for {char}")
            full = [m,n,k][vecdim]
            part = full//count
            splits = [part*j for j in range(count)]
            addr_offset_steps[i] = [sum([step for jj in range(full)],lsc_offset.zero_offset()) \
                    for j in range(count)]
            addr_offset_starts[i] = [sum([step for jj in range(j)],lsc_offset.zero_offset()) \
                    for j in splits]
            # Force split by setting max offset range to split/part
            addr_offset_ranges[i] = [(zo, sum([step for jj in range(part)],
                                              lsc_offset.zero_offset())) for \
                    j in range(count)]
        else:
            raise NotImplementedError(f"multiaddr strategy \"{strat}\" not implemented")

    print(f"addr steps: {addr_offset_steps}")
    print(f"addr starts: {addr_offset_starts}")


    ar = addr_resolver(indices=addr_indices,
                       starting_offsets=addr_offset_starts,
                       offset_ranges = addr_offset_ranges,
                       steps=addr_offset_steps)
    model = load_store_cpu(res_counts=[lsc_args.a_data_regs,
                                       lsc_args.b_data_regs,
                                       lsc_args.c_data_regs],
                           res_steps=[1,1,1],
                           ar=ar,
                           preload_counts=[lsc_args.a_preload,
                                           lsc_args.b_preload,
                                           lsc_args.c_data_regs],
                           offset_mappers=[a_mapper,b_mapper,c_mapper],
                           op=args.op)


    specializer.set_model(model=model)

    mm_ops_p1k = genmm.generate(add_dims=[0,0,0,0,0,k])
    mm_ops_p2k = genmm.generate(add_dims=[0,0,0,0,0,2*k])
    #print("\n".join(map(str,inspector(mm_ops_p1k))))

    genlog.debug("DEBUG: TRANSFORMING PRELOAD")
    preload = model.preload(ops=mm_ops,next_ops=mm_ops_p1k)
    genlog.debug("DEBUG: TRANSFORMING MAIN BLOCK")
    mainblock = model(mm_ops)
    genlog.debug("DEBUG: TRANSFORMING NEXTITER PRELOAD")
    preload_mb = model.preload(mm_ops_p1k,
                               mm_ops_p2k,
                               zero_addrs=False,
                               ignore_dims=[2])
    genlog.debug("DEBUG: TRANSFORMING STORE BLOCK")
    storeblock = model.store_modified()


    genlog.debug("################# MODIFICATIONS ##################")

    if fma_args is not None:
        if "load_bcast" == fma_args.fma_unvec_method:
            # TODO: arbitrary vecdim
            unvec_component = 1
            vec_tile = sup.a_tile
            def mod_load(op : lsc_load) -> lsc_load:
                if not isinstance(op,lsc_load):
                    return op
                if op.rtype_idx == unvec_component:
                    op.mods.add(ldst_modifier.bcast1)
                    op.tiles[1] = vec_tile

                return op
            def mod_transform(op : lsc_transformation) -> lsc_transformation:
                if not isinstance(op,lsc_transformation):
                    return op
                op.tiles[unvec_component] = vec_tile
                return op

            for mod in [mod_load,mod_transform]:
                preload = [mod(op) for op in preload]
                mainblock = [mod(op) for op in mainblock]
                preload_mb = [mod(op) for op in preload_mb]
                storeblock = [mod(op) for op in storeblock]

    lsclog.debug("################### PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,mainblock)))
    lsclog.debug("PRELOAD NEXT ----------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,preload_mb)))
    lsclog.debug("END MAIN LOOP ---------------------------")
    lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,storeblock)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")

    specializer.analyse(preload)
    specializer.analyse(mainblock)
    specializer.analyse(storeblock)
    specializer.analyse(preload_mb)

    preload = specializer.pre_specialize(ops=preload, triple=triple)
    mainblock = specializer.pre_specialize(ops=mainblock, triple=triple)
    storeblock = specializer.pre_specialize(ops=storeblock, triple=triple)
    preload_mb = specializer.pre_specialize(ops=preload_mb, triple=triple)

    lsclog.debug("################### SPECIALIZED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,mainblock)))
    lsclog.debug("PRELOAD NEXT ----------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,preload_mb)))
    lsclog.debug("END MAIN LOOP ---------------------------")
    lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,storeblock)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")


    sched_args,rest = parse_sched_args(parser=parser)

    
    scheduler = simple_dependency_scheduler(
            rar=sched_args.sched_rar_distance,
            raw=sched_args.sched_raw_distance,
            war=sched_args.sched_war_distance,
            waw=sched_args.sched_waw_distance,
            debug_on=False)

    rs_preload = scheduler(preload, loop=False)
    rs_mbpl = scheduler(mainblock+preload_mb)
    # will fail with rvv
    rs_store = storeblock
    #rs_store = scheduler(storeblock, loop=False)


    lsclog.debug("################### RE-SCHEDULED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,rs_preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,rs_mbpl)))
    lsclog.debug("END MAIN LOOP ---------------------------")
    lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,rs_store)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")

    initblock = specializer.code_init(triple=triple)

    asm_rs_preload = specializer.specialize(
            ops=[op.op for op in rs_preload],
            triple=triple)
    asm_rs_mbpl = specializer.specialize(
            ops=[op.op for op in rs_mbpl],
            triple=triple)
    asm_rs_store = specializer.specialize(
            ops=rs_store,
            triple=triple)



    finiblock = specializer.code_fini(triple=triple)

    asmlog.debug("################### ASM ###################")
    asmlog.debug("INIT ------------------------------------")
    asmlog.debug(initblock)
    asmlog.debug("PRELOAD ---------------------------------")
    asmlog.debug("".join(asm_rs_preload))
    asmlog.debug("MAIN LOOP -------------------------------")
    asmlog.debug("  "+"  ".join(asm_rs_mbpl))
    asmlog.debug("END MAIN LOOP ---------------------------")
    asmlog.debug("STOREBLOCK ------------------------------")
    asmlog.debug("".join(asm_rs_store))
    asmlog.debug("ENDSTOREBLOCK ---------------------------")
    asmlog.debug("FINALIZE --------------------------------")
    asmlog.debug(finiblock)


    genlog.debug("Aliased GP regs in the end:")
    for alias,regidx in rt.aliased_regs['greg'].items():
        genlog.debug(f"  {alias:30} : {gen.greg(regidx)}")


    inout_args,rest = parse_inout_args(parser=parser)

    # TODO: handle help in a better way
    # This was the last parser so check for help string
    if any(a in rest for a in helpargs):
        if [a for a in rest if a not in helpargs]:
            print(f"uknown arguments: {rest}")
        parser.print_help()
        sys.exit(0)


    tpl_data = ""

    genlog.debug(f"Reading source template from {inout_args.input_filename}")
    with open(inout_args.input_filename, 'r') as file:
        tpl_data = file.read()

    genlog.debug(f"Writing source to {inout_args.output_filename}")
    with open(inout_args.output_filename, 'w') as file:
        file.write(tpl_data)
    


if __name__ == "__main__":
     main()
