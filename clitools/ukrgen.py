# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import argparse
import sys
import logging

from copy import deepcopy

from mako.template import Template

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.asmblocks.operations import widening_method as wm
from asmgen.asmblocks.noarch import comparison

from asmgen.registers import (
        asm_data_type as adt,
        adt_triple,
        adt_size,
        reg_tracker
        )

from asmgen.callconv.callconv import callconv
from ukrgen.models.loop import lsc_comparison,lsc_condition,lsc_loop

from ukrgen.specializers.asm import lsc_specializer
from ukrgen.components import tile,simple_ukr_tile,dimension_type,dimension_properties
from ukrgen.components.tile import scalar_dp
from ukrgen.generators.mm import mm,order2D
from ukrgen.models.load_store_cpu import load_store_cpu
from ukrgen.models.load_store_operations import (
    lsc_add_val_off,
    lsc_load,
    lsc_transformation,
    ldst_modifier
)
from ukrgen.models.lsc.offset import lsc_offset,stridexvlen
from ukrgen.models.addr_resolver import addr_resolver
from ukrgen.models.offset_mapper import (
        flat_mapper,
        strided_mapper,
        same_address_mapper
    )
from ukrgen.schedulers import simple_dependency_scheduler,minreguse_scheduler

from .internal.addr_parameters import calculate_addr_parameters
from .internal.ukr import get_ukr_components
from .internal.compose_mm import fngen,get_blis_gemm_cc

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

    kernels = ['mm','gemm']
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

def parse_mm_args(parser : argparse.ArgumentParser):


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

def parse_stride_args(parser : argparse.ArgumentParser, ukr : str):

    choices = ['A','B','C']
    if "gemm" == ukr:
        choices =['A','B','AB','C']


    parser.add_argument("--column-strides", type=str, nargs="+", required=False,
                        choices=choices,
                        help="components with general column strides")
    parser.add_argument("--row-strides", type=str, nargs="+", required=False,
                        choices=choices,
                        help="components with general row strides")

    args, rest = parser.parse_known_args()
    helpexit_if_last_parser(rest=rest, parser=parser)
    return args,rest

def parse_lsc_args(parser : argparse.ArgumentParser, ukr : str):

    components = ['A','B','C']
    if "gemm" == ukr:
        components =['A','B','AB','C']


    multiareg_strategies = ["interleave","split","phase"]

    default_mars = {
        "A" : "interleave",
        "B" : "interleave",
        "AB" : "phase",
        "C" : "phase",
    }

    for c in components:
        parser.add_argument(f"--{c}-data-regs", type=int, required=True,
                            help=f"Number of data registers to use for the {c} component")
        parser.add_argument(f"--{c}-addr-regs", type=int,
                            default=1,
                            help=f"number of address registers to use for the {c} component")
        parser.add_argument(f"--{c}-multiaddr-strat", type=str,
                            default=default_mars[c],
                            choices=multiareg_strategies,
                            help=f"Strategy for using multiple address registers for {c} component")
        if c in ["A","B"]:
            parser.add_argument(f"--{c}-preload", type=int, required=True,
                                help=f"number of {c} data registers to preload")

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
                        help="Path to the file to output the generated ASM function to")

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
    
    fngenlog = logging.getLogger("COMPOSER")
    fngenlog.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
            description="ukrgen compute kernel generator",
            add_help=False)

    args,rest = parse_main_arguments(parser=parser)

    gen = asmgen_map[args.isa]()

    gen.set_output_inline(yesno=False)

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

    component_dts = {
        "A" : triple.a,
        "B" : triple.b,
        "AB" : triple.c,
        "C" : triple.c,
        "alpha" : triple.c,
        "beta" : triple.c
    }

    ways = adt_size(triple.c)//adt_size(triple.a)


    #TODO: other kernels
    parse_ukr_args = parse_mm_args
    if args.ukr not in ["gemm","mm"]:
        raise NotImplementedError("parsing non-mm args not implemented")

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

    if "mm" == args.ukr:
        genmm = mm(a=a_tile, b=b_tile, c=c_tile, lo=order, opstr=args.op,
                   tile_strs=["A","B","C"])
    if "gemm" == args.ukr:
        genmm = mm(a=a_tile, b=b_tile, c=c_tile, lo=order, opstr=args.op,
                   tile_strs=["A","B","AB"])

        scale_tile = simple_ukr_tile(a_size=m,
                                     b_size=n,
                                     subdims=(sup.c_tile.dima,
                                              sup.c_tile.dimb))
        alphabeta_tile = simple_ukr_tile(a_size=n, b_size=n,
                                    subdims=(scalar_dp,scalar_dp),
                                    bands=(0,0))
        genbetascale = mm(scale_tile, alphabeta_tile, scale_tile,
                          opstr="fmul", tile_strs=["C","beta","C"])
        genalphascale = mm(scale_tile, alphabeta_tile, scale_tile,
                        opstr="fma", tile_strs=["AB","alpha","C"])
        betascale_ops = genbetascale.generate()
        alphascale_ops = genalphascale.generate()

    mm_ops = genmm.generate()


    tiflog.debug("################### TILE-INSTRUCTION-FORMAT ###################")
    tiflog.debug("### MAIN ###")
    for op in mm_ops:
        tiflog.debug(str(op))
    tiflog.debug("### BETA SCALE ###")

    if "gemm" == args.ukr:
        for op in betascale_ops:
            tiflog.debug(str(op))
        tiflog.debug("### ALPHA SCALE ###")
        for op in alphascale_ops:
            tiflog.debug(str(op))

    stride_args,rest = parse_stride_args(parser=parser, ukr=args.ukr)

    lsc_args,rest = parse_lsc_args(parser=parser, ukr=args.ukr)

    components = get_ukr_components(args.ukr)

    addr_indices = { c :[i for i in range(lsc_args.__dict__[f"{c}_addr_regs"])]
                    for c in components }

    addr_reg_counts = { c : lsc_args.__dict__[f"{c}_addr_regs"]
                    for c in components }

    data_reg_counts = { c : lsc_args.__dict__[f"{c}_data_regs"]
                       for c in components}

    if "gemm" == args.ukr:
        addr_indices["alpha"] = [0]
        addr_indices["beta"] = [0]
        data_reg_counts["alpha"] = 1
        data_reg_counts["beta"] = 1
        addr_reg_counts["alpha"] = 1
        addr_reg_counts["beta"] = 1


    strides = {k : (None,None) for k in components}
    mappers = {}
    i = 0
    for char in components:
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

    mappers['A'] = strided_mapper((m,1), strides['A'],vecdim=0)
    mappers['B'] = strided_mapper((1,n), strides['B'],vecdim=1)
    mappers['C'] = strided_mapper((m,n), strides['C'],vecdim=0)

    if "gemm" == args.ukr:
        mappers['AB'] = strided_mapper((m,n), strides['AB'],vecdim=0)

    scalar_mapper = same_address_mapper()

    mappers['beta'] = scalar_mapper
    mappers['alpha'] = scalar_mapper


    scalar_tile = tile(dima=scalar_dp, dimb=scalar_dp)

    component_tiles = {
        "A" : sup.a_tile,
        "B" : sup.b_tile,
        "C" : sup.c_tile
    }
    if "gemm" == args.ukr:
        component_tiles = component_tiles | {
        "AB" : sup.c_tile,
        "beta" : scalar_tile,
        "alpha" : scalar_tile
    }

    real_tiles = deepcopy(component_tiles)
    if args.op == 'fma' and fma_args is not None:
        if fma_args.fma_unvec_method in ['load_bcast']:
            real_tiles["B"] = real_tiles["A"]

    strats = { c : lsc_args.__dict__[f"{c}_multiaddr_strat"] \
            for c in components}

    strats['beta'] = "interleave"
    strats['alpha'] = "interleave"


    off_starts,off_ranges,off_steps = calculate_addr_parameters(
            sup=sup, primary_op=args.op, gen=gen,
            addr_reg_counts=addr_reg_counts,
            data_tiles=component_tiles, 
            real_tiles=real_tiles, 
            narrow_components=["A","B"], 
            wide_components=["AB","C","beta","alpha"],
            m=m,n=n,k=k,
            vecdims={
                "A" : 0,
                "B" : 1,
                "C" : 0,
                "AB" : 0,
                "beta" : 0,
                "alpha" : 0,
                },
            strides=strides,
            mappers=mappers,
            strats=strats)

    print(f"addr steps: {off_steps}")
    print(f"addr starts: {off_starts}")


    ar = addr_resolver(indices          = addr_indices,
                       starting_offsets = off_starts,
                       offset_ranges    = off_ranges,
                       steps            = off_steps,
                       max_incs=4)


    #TODO: investigate if there are architectures where this is relevant
    res_steps = {c:1 for c in components}

    preload_counts = {c : lsc_args.__dict__[f"{c}_preload"] for c in ["A","B"]}
    preload_counts["C"] = lsc_args.C_data_regs
    if "gemm" == args.ukr:
        preload_counts["AB"] = lsc_args.C_data_regs
        preload_counts["beta"] = 0
        preload_counts["alpha"] = 0
        res_steps["alpha"] = 1
        res_steps["beta"] = 1


    resolve_order = deepcopy(components)
    if "gemm" == args.ukr:
        resolve_order.extend(["beta","alpha"])

    model = load_store_cpu(res_counts=data_reg_counts,
                           res_steps=res_steps,
                           ar=ar,
                           preload_counts=preload_counts,
                           offset_mappers=mappers,
                           #TODO: parameterize resolve order
                           resolve_order=resolve_order)


    specializer.set_model(model=model)

    mm_ops_p1k = genmm.generate(add_dims=[0,0,0,0,0,k])
    mm_ops_p2k = genmm.generate(add_dims=[0,0,0,0,0,2*k])
    #print("\n".join(map(str,inspector(mm_ops_p1k))))

    genlog.debug("DEBUG: TRANSFORMING PRELOAD")
    preload = model.preload(ops=mm_ops,next_ops=mm_ops_p1k,
                            zero_components=["C","AB"],
                            ignore_components=[])
    genlog.debug("DEBUG: TRANSFORMING MAIN BLOCK")
    mainblock = model(mm_ops)

    genlog.debug("DEBUG: TRANSFORMING NEXTITER PRELOAD")
    preload_mb = model.preload(mm_ops_p1k,
                               mm_ops_p2k,
                               zero_addrs=False,
                               zero_components=[],
                               ignore_components=["C","AB"])

    if "gemm" == args.ukr:
        genlog.debug("DEBUG: TRANSFORMING BETASCALE BLOCK")
        betascale = model(betascale_ops)
        genlog.debug("DEBUG: TRANSFORMING ALPHASCALE BLOCK")
        alphascale = model(alphascale_ops)
    genlog.debug("DEBUG: TRANSFORMING STORE BLOCK")
    storeblock = model.store_modified(ignore_components="AB")


    genlog.debug("################# MODIFICATIONS ##################")

    if fma_args is not None:
        # TODO: arbitrary vecdim
        unvec_components = ["B","alpha","beta"]
        if "load_bcast" == fma_args.fma_unvec_method:
            vec_tile = sup.a_tile
            def mod_load(op : lsc_load) -> lsc_load:
                if not isinstance(op,lsc_load):
                    return op
                if op.addr_idx.component in unvec_components:
                    op.mods.add(ldst_modifier.bcast1)
                    op.tiles[1] = vec_tile

                return op
            def mod_transform(op : lsc_transformation) -> lsc_transformation:
                if not isinstance(op,lsc_transformation):
                    return op
                for i,idx in enumerate(op.indices):
                    c = idx.component
                    if c in unvec_components:
                        op.tiles[i] = vec_tile

                return op

            for mod in [mod_load,mod_transform]:
                preload = [mod(op) for op in preload]
                mainblock = [mod(op) for op in mainblock]
                preload_mb = [mod(op) for op in preload_mb]
                if "gemm" == args.ukr:
                    betascale = [mod(op) for op in betascale]
                    alphascale = [mod(op) for op in alphascale]
                storeblock = [mod(op) for op in storeblock]

    lsclog.debug("################### PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,mainblock)))
    lsclog.debug("PRELOAD NEXT ----------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,preload_mb)))
    lsclog.debug("END MAIN LOOP ---------------------------")

    if "gemm" == args.ukr:
        lsclog.debug("BETASCALE BLOCK -------------------------")
        lsclog.debug("\n".join(map(str,betascale)))
        lsclog.debug("END BETASCALE BLOCK ---------------------")
        lsclog.debug("ALPHASCALE BLOCK ------------------------")
        lsclog.debug("\n".join(map(str,alphascale)))
        lsclog.debug("END ALPHASCALE BLOCK --------------------")
    lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,storeblock)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")

    specializer.analyse(preload)
    specializer.analyse(mainblock)
    if "gemm" == args.ukr:
        specializer.analyse(betascale)
        specializer.analyse(alphascale)
    specializer.analyse(storeblock)
    specializer.analyse(preload_mb)

    preload = specializer.pre_specialize(ops=preload, component_dts=component_dts)
    mainblock = specializer.pre_specialize(ops=mainblock, component_dts=component_dts)
    if "gemm" == args.ukr:
        betablock = specializer.pre_specialize(ops=betascale, component_dts=component_dts)
        alphablock = specializer.pre_specialize(ops=alphascale, component_dts=component_dts)
    storeblock = specializer.pre_specialize(ops=storeblock, component_dts=component_dts)
    preload_mb = specializer.pre_specialize(ops=preload_mb, component_dts=component_dts)

    lsclog.debug("################### SPECIALIZED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,mainblock)))
    lsclog.debug("PRELOAD NEXT ----------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,preload_mb)))
    lsclog.debug("END MAIN LOOP ---------------------------")
    if "gemm" == args.ukr:
        lsclog.debug("BETASCALE BLOCK -------------------------")
        lsclog.debug("\n".join(map(str,betablock)))
        lsclog.debug("END BETASCALE BLOCK ---------------------")
        lsclog.debug("ALPHASCALE BLOCK ------------------------")
        lsclog.debug("\n".join(map(str,alphablock)))
        lsclog.debug("END ALPHASCALE BLOCK --------------------")
    lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,storeblock)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")


    sched_args,rest = parse_sched_args(parser=parser)

    # register reuse 
    mru_scheduler = minreguse_scheduler()

    mru_scheduler.analyze_preceeding(preload+mainblock+preload_mb)
    premru_storeblock = storeblock
    if "gemm" == args.ukr:
        premru_storeblock = betablock+alphablock+storeblock

    mru_scheduler.analyze(premru_storeblock)
    premru_storeblock = mru_scheduler.reorder(premru_storeblock)
    lsclog.debug("################### MRU STORE REORDERED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,premru_storeblock)))
    mru_scheduler.analyze(premru_storeblock)
    mru_storeblock = mru_scheduler.replace(premru_storeblock,
                                           specializer.data_registers)
    
    lsclog.debug("################### MRU STORE RENAMED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,mru_storeblock)))


    #sys.exit(0)

    
    scheduler = simple_dependency_scheduler(
            rar=sched_args.sched_rar_distance,
            raw=sched_args.sched_raw_distance,
            war=sched_args.sched_war_distance,
            waw=sched_args.sched_waw_distance,
            debug_on=False)

    rs_preload = scheduler(preload, loop=False)
    rs_mbpl = scheduler(mainblock+preload_mb)

    rs_store = scheduler(mru_storeblock)
    #if "gemm" == args.ukr:
    #    rs_store = scheduler(betablock+alphablock+storeblock)
    #else:
    #    rs_store = storeblock
    #rs_store = scheduler(storeblock, loop=False)


    lsclog.debug("################### RE-SCHEDULED PSEUDO-ASM ###################")
    lsclog.debug("\n".join(map(str,rs_preload)))
    lsclog.debug("MAIN LOOP -------------------------------")
    lsclog.debug("  "+"\n  ".join(map(str,rs_mbpl)))
    lsclog.debug("END MAIN LOOP ---------------------------")

    if "gemm" == args.ukr:
        lsclog.debug("SCALE+STOREBLOCK ------------------------")
    else:
        lsclog.debug("STOREBLOCK ------------------------------")
    lsclog.debug("\n".join(map(str,rs_store)))
    lsclog.debug("ENDSTOREBLOCK ---------------------------")





    gemm_fngen = fngen(gen=gen, rt=rt)


    stride_map : dict[str,str] = dict()
    # gotta map {r,c}s_{a,b,c} onto strideN
    for component,rcs in strides.items():
        if rcs[0] is not None:
            stride_map[f"rs_{component}"] = f"stride{rcs[0]}"
        if rcs[1] is not None:
            stride_map[f"cs_{component}"] = f"stride{rcs[1]}"

    blis_cc = get_blis_gemm_cc(gen=gen)
    gemm_fngen.init_cc(cc=blis_cc,
                       reverse_alias_map=stride_map)

    # Add the loop

    condition = lsc_condition(first="k", second=None, 
                              comparison=lsc_comparison(comparison.nz))
    mainloop = lsc_loop(name="knloop", condition=condition, level=2)

    mainloop.add_block(rs_mbpl)
    mainloop.add_block(ops=[
        lsc_add_val_off("k", off=lsc_offset({},[],[],-1))
        ])

    # TODO: figure out prefetching
    #if distance_pfc:
    #    vecdim=0
    #    count = [m,n][vecdim]
    #    for i in m:
    #        asmblock += ""
    #    loop.add_singleshot_divergence(name="pfc", block="")



    initblock = specializer.code_init(component_dts=component_dts)

    asm_rs_preload = specializer.specialize(
            ops=rs_preload,
            component_dts=component_dts)
    asm_rs_mainloop = specializer.specialize(
            ops=[mainloop],
            component_dts=component_dts)
    asm_rs_store = specializer.specialize(
            ops=rs_store,
            component_dts=component_dts)



    finiblock = specializer.code_fini(component_dts=component_dts)


    fnsave,fnload,fnrestore = gemm_fngen.get_boilerplate(cc=blis_cc)


    asmblock = gen.asmwrap(
        "# FUNC INTRO ---------------------------------")
    asmblock += fnsave
    asmblock += fnload
    asmblock += gen.asmwrap(
        "# INIT ---------------------------------------")
    asmblock += initblock
    asmblock += gen.asmwrap(
        "# PRELOAD ------------------------------------")
    asmblock += "".join(asm_rs_preload)
    asmblock += gen.asmwrap(
        "# MAIN LOOP ----------------------------------")
    asmblock += "  "+"  ".join(asm_rs_mainloop)
    asmblock += gen.asmwrap(
        "# END MAIN LOOP ------------------------------")
    if "gemm" == args.ukr:
        asmblock += gen.asmwrap(
            "# SCALE+STOREBLOCK ---------------------------")
    else:
        asmblock += gen.asmwrap(
            "# STOREBLOCK ---------------------------------")
    asmblock += "".join(asm_rs_store)
    asmblock += gen.asmwrap(
        "# END STOREBLOCK -----------------------------")
    asmblock += gen.asmwrap(
        "# FINALIZE -----------------------------------")
    asmblock += finiblock
    asmblock += gen.asmwrap(
        "# FUNC OUTRO ---------------------------------")
    asmblock += fnrestore
    asmblock += gen.asmwrap("ret") # TODO: add ret() to asmgen


    asmlog.debug("################### ASM ###################")
    asmlog.debug("FUNC INTRO ------------------------------")
    asmlog.debug(fnsave)
    asmlog.debug(fnload)
    asmlog.debug("INIT ------------------------------------")
    asmlog.debug(initblock)
    asmlog.debug("PRELOAD ---------------------------------")
    asmlog.debug("".join(asm_rs_preload))
    asmlog.debug("MAIN LOOP -------------------------------")
    asmlog.debug("  "+"  ".join(asm_rs_mainloop))
    asmlog.debug("END MAIN LOOP ---------------------------")
    if "gemm" == args.ukr:
        lsclog.debug("SCALE+STOREBLOCK ------------------------")
    else:
        lsclog.debug("STOREBLOCK ------------------------------")
    asmlog.debug("".join(asm_rs_store))
    asmlog.debug("ENDSTOREBLOCK ---------------------------")
    asmlog.debug("FINALIZE --------------------------------")
    asmlog.debug(finiblock)
    asmlog.debug("FUNC OUTRO ------------------------------")
    asmlog.debug(fnrestore)



    genlog.debug("Aliased GP regs in the end:")
    for alias,regidx in rt.aliased_regs['greg'].items():
        genlog.debug(f"  {alias:30} : {gen.greg(regidx)}")

    genlog.debug("Aliased VEC regs in the end:")
    for alias,regidx in rt.aliased_regs['vreg'].items():
        genlog.debug(f"  {alias:30} : {gen.vreg(regidx)}")



    inout_args,rest = parse_inout_args(parser=parser)

    # TODO: handle help in a better way
    # This was the last parser so check for help string
    if any(a in rest for a in helpargs):
        if [a for a in rest if a not in helpargs]:
            print(f"uknown arguments: {rest}")
        parser.print_help()
        sys.exit(0)


    

    asmheader = (
         ".section .text\n"
        f".global gemm_kernel_{m}Vx{n}\n"
        f"gemm_kernel_{m}Vx{n}:\n"
    )

    genlog.debug(f"Writing source to {inout_args.output_filename}")
    with open(inout_args.output_filename, 'w') as file:
        file.write(asmheader+asmblock)


if __name__ == "__main__":
     main()
