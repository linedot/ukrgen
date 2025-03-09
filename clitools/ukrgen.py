import argparse

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve


from asmgen.registers import asm_data_type as adt
from algobuild.specializers.asm import lsc_specializer


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

    spec_args, rest = parser.parse_known_args()

    ab_type = adt[spec_args.ab_data_type]
    c_type = adt[spec_args.c_data_type]
    print(f"Operation support for {op} with A/B={spec_args.ab_data_type} and C={spec_args.c_data_type}:")
    for sup in specializer.op_support_map[op]:
        if ab_type == sup.triple.a and \
           ab_type == sup.triple.b and \
           c_type == sup.triple.c:
               print(sup)

def main():

    args,rest = parse_main_arguments()

    specializer = lsc_specializer(model=None, gen=asmgen_map[args.isa](), rt=None)

    #print("Generator supports following operations:")
    #for k,v in specializer.op_support_map.items():
    #    print(f"Operation {k}:")
    #    for sup in v:
    #        print(f"{sup}")
    #print(f"Operation support for {args.op}:")
    #for sup in specializer.op_support_map[args.op]:
    #    print(f"{sup}")

    spec_args = parse_specializer_arguments(specializer=specializer, op=args.op, args=rest)

if __name__ == "__main__":
     main()
