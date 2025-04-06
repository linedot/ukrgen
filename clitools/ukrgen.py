import argparse

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.asmblocks.operations import widening_method as wm


from asmgen.registers import asm_data_type as adt,adt_triple,adt_size
from algobuild.specializers.asm import lsc_specializer
from algobuild.components import simple_ukr_tile,dimension_type,dimension_properties
from algobuild.generators.mm import mm,order2D
from algobuild.models.load_store_cpu import load_store_cpu
from algobuild.schedulers import simple_dependency_scheduler


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

def parse_main_arguments():

    operations = ['fma','dota','fopa','mma']

    parser = argparse.ArgumentParser(description="ukrgen compute kernel generator")
    parser.add_argument("--isa", type=str,
                        choices=asmgen_map.keys(),
                        required=True, help="ISA to use")
    parser.add_argument("--op", type=str,
                        choices=operations,
                        required=True, help="Arithmetic operation")

    return parser.parse_known_args()


def parse_specializer_arguments(specializer : lsc_specializer, op : str, args : list[str]):

    parser = argparse.ArgumentParser(description="Arguments for kernel specialization")
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


    spec_args, rest = parser.parse_known_args(args=args)

    

    ab_type = adt[spec_args.ab_data_type]
    c_type = adt[spec_args.c_data_type]

    op_support_map = [sup for sup in specializer.op_support_map[op] if \
            (ab_type == sup.triple.a and \
             ab_type == sup.triple.b and \
             c_type == sup.triple.c)
            ]

    parser.add_argument("--variant", type=int, required=True,
                        choices=list(range(len(op_support_map))),
                        help="Variant to use")

    spec_args, rest = parser.parse_known_args(args=args)


    return spec_args, rest, op_support_map[spec_args.variant]

def parse_fma_args(args : list[str]):

    parser = argparse.ArgumentParser(description="Arguments for fma instruction")

    parser.add_argument("--fma-b-method", type=str, required=True,
                        choices=['bcast1','idx','bcastidx'],
                        help="How to access elements in b vector for the fma")

    return parser.parse_known_args(args=args)

def parse_ukr_args(args : list[str]):

    parser = argparse.ArgumentParser(description="Arguments for kernel generation")

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

    return parser.parse_known_args(args=args)

def parse_lsc_args(args : list[str]):
    parser = argparse.ArgumentParser(description="Arguments for the lsc machine model")

    parser.add_argument("--a-data-regs", type=int, required=True,
                        help="number of data registers to use for the a tiles")
    parser.add_argument("--b-data-regs", type=int, required=True,
                        help="number of data registers to use for the b tiles")
    parser.add_argument("--c-data-regs", type=int, required=True,
                        help="number of data registers to use for the c tiles")
    parser.add_argument("--a-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the a tiles")
    parser.add_argument("--b-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the b tiles")
    parser.add_argument("--c-addr-regs", type=int,
                        default=1,
                        help="number of address registers to use for the c tiles")
    parser.add_argument("--a-preload", type=int, required=True,
                        help="number of a data registers to preload")
    parser.add_argument("--b-preload", type=int, required=True,
                        help="number of b data registers to preload")

    return parser.parse_known_args(args=args)

def parse_sched_args(args : list[str]):
    parser = argparse.ArgumentParser(description="Arguments for the lsc scheduler")

    parser.add_argument("--sched-rar-distance", type=int, default=0,
                        help="Minimum read-after-read distance")
    parser.add_argument("--sched-raw-distance", type=int, default=0,
                        help="Minimum read-after-write distance")
    parser.add_argument("--sched-war-distance", type=int, default=0,
                        help="Minimum write-after-read distance")
    parser.add_argument("--sched-waw-distance", type=int, default=0,
                        help="Minimum write-after-waw distance")

    return parser.parse_known_args(args=args)

def main():

    args,rest = parse_main_arguments()

    gen = asmgen_map[args.isa]()

    specializer = lsc_specializer(model=None, gen=gen, rt=None)

    #print("Generator supports following operations:")
    #for k,v in specializer.op_support_map.items():
    #    print(f"Operation {k}:")
    #    for sup in v:
    #        print(f"{sup}")
    #print(f"Operation support for {args.op}:")
    #for sup in specializer.op_support_map[args.op]:
    #    print(f"{sup}")


    spec_args,rest,sup = parse_specializer_arguments(specializer=specializer, op=args.op, args=rest)


    triple = adt_triple(a_dt=adt[spec_args.ab_data_type],
                        b_dt=adt[spec_args.ab_data_type],
                        c_dt=adt[spec_args.c_data_type])

    ways = adt_size(triple.c)//adt_size(triple.a)


    ukr_args,rest = parse_ukr_args(args=rest)

    m,n,k = ukr_args.m,ukr_args.n,ukr_args.k
    order = order2D(ukr_args.order)

    nc=n
    nb=n
    if args.op == 'fma' and sup.b_tile.dima == sup.a_tile.dima:
        fma_args, rest = parse_fma_args(args=rest)
        if 'bcast1' == fma_args.fma_b_method:
            sup.b_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
        elif fma_args.fma_b_method in ['idx','bcastidx']:
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

    mm_ops = genmm.generate()

    print("################### TILE-INSTRUCTION-FORMAT ###################")
    for op in mm_ops:
        print(str(op))

    lsc_args,rest = parse_lsc_args(args=rest)

    addr_offset_ranges=[
        [(0,gen.max_load_voff) for i in range(lsc_args.a_addr_regs)], 
        [(0,gen.max_load_voff) for i in range(lsc_args.b_addr_regs)], 
        [(0,gen.max_load_voff) for i in range(lsc_args.c_addr_regs)], 
    ]
    if sup.a_tile.dima.dt == dimension_type.fixed and sup.a_tile.dima.size == 1:
        addr_offset_ranges[0] = [(0,gen.max_fload_immoff(dt=sup.triple.a)) for i in range(lsc_args.a_addr_regs)]
    if sup.b_tile.dima.dt == dimension_type.fixed and sup.b_tile.dima.size == 1:
        addr_offset_ranges[1] = [(0,gen.max_fload_immoff(dt=sup.triple.b)) for i in range(lsc_args.b_addr_regs)]
    if sup.c_tile.dima.dt == dimension_type.fixed and sup.c_tile.dima.size == 1:
        addr_offset_ranges[2] = [(0,gen.max_fload_immoff(dt=sup.triple.c)) for i in range(lsc_args.c_addr_regs)]


    # Ensure the specializer doesn't generate impossible voffsets for loads/stores of
    # C regs
    if getattr(gen, args.op).widening_method == wm.SPLIT_INSTRUCTIONS:
        for i in range(len(addr_offset_ranges[2])):
            addr_offset_ranges[2][i] = (addr_offset_ranges[2][i][0],addr_offset_ranges[2][i][1]//ways)


    ac_mapper = lambda tile, idx : tile.dima.size*m*idx[1]+idx[0]
    b_mapper = lambda tile, idx : tile.dima.size*n*idx[0]+idx[1]

    model = load_store_cpu(res_counts=[lsc_args.a_data_regs,
                                       lsc_args.b_data_regs,
                                       lsc_args.c_data_regs],
                           res_steps=[1,1,1],
                           addr_counts=[lsc_args.a_addr_regs,
                                       lsc_args.b_addr_regs,
                                       lsc_args.c_addr_regs],
                           addr_offset_ranges=addr_offset_ranges,
                           addr_starts=[
                               [i*sup.a_tile.dima.size for i in range(lsc_args.a_addr_regs)],
                               [i*sup.b_tile.dima.size for i in range(lsc_args.b_addr_regs)],
                               [i*sup.c_tile.dima.size for i in range(lsc_args.c_addr_regs)]
                               ],
                           preload_counts=[lsc_args.a_preload,
                                           lsc_args.b_preload,
                                           lsc_args.c_data_regs],
                           offset_mappers=[ac_mapper,b_mapper,ac_mapper])

    mm_ops_next = genmm.generate(add_dims=[0,0,0,0,0,k])
    #print("\n".join(map(str,inspector(mm_ops_next))))

    preload = model.preload(mm_ops)
    mainblock = model(mm_ops)
    storeblock = model.store_modified()
    preload_mb = model.preload(mm_ops_next,
                                   zero_addrs=False,
                                   ignore_dims=[2])

    print("################### PSEUDO-ASM ###################")
    print("\n".join(map(str,preload)))
    print("MAIN LOOP -------------------------------")
    print("  "+"\n  ".join(map(str,mainblock)))
    print("PRELOAD NEXT ----------------------------")
    print("  "+"\n  ".join(map(str,preload_mb)))
    print("END MAIN LOOP ---------------------------")
    print("STOREBLOCK ------------------------------")
    print("  "+"\n  ".join(map(str,storeblock)))
    print("ENDSTOREBLOCK ---------------------------")

    specializer.analyse(preload)
    specializer.analyse(mainblock)
    specializer.analyse(storeblock)
    specializer.analyse(preload_mb)

    preload = specializer.pre_specialize(ops=preload, triple=triple)
    mainblock = specializer.pre_specialize(ops=mainblock, triple=triple)
    storeblock = specializer.pre_specialize(ops=storeblock, triple=triple)
    preload_mb = specializer.pre_specialize(ops=preload_mb, triple=triple)

    print("################### SPECIALIZED PSEUDO-ASM ###################")
    print("\n".join(map(str,preload)))
    print("MAIN LOOP -------------------------------")
    print("  "+"\n  ".join(map(str,mainblock)))
    print("PRELOAD NEXT ----------------------------")
    print("  "+"\n  ".join(map(str,preload_mb)))
    print("END MAIN LOOP ---------------------------")
    print("STOREBLOCK ------------------------------")
    print("  "+"\n  ".join(map(str,storeblock)))
    print("ENDSTOREBLOCK ---------------------------")


    sched_args,rest = parse_sched_args(args=rest)
    
    scheduler = simple_dependency_scheduler(
            rar=sched_args.sched_rar_distance,
            raw=sched_args.sched_raw_distance,
            war=sched_args.sched_war_distance,
            waw=sched_args.sched_waw_distance,
            debug_on=True)

    rs_preload = scheduler(preload, loop=False)
    rs_mbpl = scheduler(mainblock+preload_mb)
    # will fail with rvv
    rs_store = storeblock
    #rs_store = scheduler(storeblock, loop=False)


    print("################### RE-SCHEDULED PSEUDO-ASM ###################")
    print("\n".join(map(str,rs_preload)))
    print("MAIN LOOP -------------------------------")
    print("  "+"\n  ".join(map(str,rs_mbpl)))
    print("END MAIN LOOP ---------------------------")
    print("STOREBLOCK ------------------------------")
    print("  "+"\n  ".join(map(str,rs_store)))
    print("ENDSTOREBLOCK ---------------------------")

if __name__ == "__main__":
     main()
