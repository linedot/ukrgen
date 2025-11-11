# ukrgen - compute kernel generators based on asmgen

## New generator

WIP

Usage:
```
usage: ukrgen.py --ukr {mm,gemm}
                 --isa {avx128,avx256,avx512,rvv,rvv071,neon,sve,sme}
                 --op {fma,dota,fopa,mma}
                 --ab-data-type {HALF,HALF,HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT8,UINT8,SINT8,SINT8,SINT16,SINT16,UINT16,UINT16,SINT16,SINT16,SINT32,SINT32,UINT32,UINT32,SINT32,SINT32,SINT64,SINT64}
                 --c-data-type {HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT16,UINT16,SINT16,SINT16,SINT16,SINT16,UINT32,UINT32,SINT32,SINT32,SINT32,SINT32,UINT64,UINT64,SINT64,SINT64,SINT64,SINT64}
                 --variant {0,1} --m M --n N [--k K] [--order ORDER]
                 [--column-strides {A,B,AB,C} [{A,B,AB,C} ...]]
                 [--row-strides {A,B,AB,C} [{A,B,AB,C} ...]]
                 --A-data-regs A_DATA_REGS [--A-addr-regs A_ADDR_REGS]
                 [--A-multiaddr-strat {interleave,split,phase}]
                 --A-preload A_PRELOAD --B-data-regs B_DATA_REGS
                 [--B-addr-regs B_ADDR_REGS]
                 [--B-multiaddr-strat {interleave,split,phase}]
                 --B-preload B_PRELOAD --AB-data-regs AB_DATA_REGS
                 [--AB-addr-regs AB_ADDR_REGS]
                 [--AB-multiaddr-strat {interleave,split,phase}]
                 --C-data-regs C_DATA_REGS [--C-addr-regs C_ADDR_REGS]
                 [--C-multiaddr-strat {interleave,split,phase}]
                 [--sched-rar-distance SCHED_RAR_DISTANCE]
                 [--sched-raw-distance SCHED_RAW_DISTANCE]
                 [--sched-war-distance SCHED_WAR_DISTANCE]
                 [--sched-waw-distance SCHED_WAW_DISTANCE]
                 --output-filename OUTPUT_FILENAME [--function-name FUNCTION_NAME]

ukrgen compute kernel generator

options:
  --ukr {mm,gemm}       Kernel to generate
  --isa {avx128,avx256,avx512,rvv,rvv071,neon,sve,sme}
                        ISA to use
  --op {fma,dota,fopa,mma}
                        Arithmetic operation
  --ab-data-type {HALF,HALF,HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT8,UINT8,SINT8,SINT8,SINT16,SINT16,UINT16,UINT16,SINT16,SINT16,SINT32,SINT32,UINT32,UINT32,SINT32,SINT32,SINT64,SINT64}
                        Type to use for A/B tiles
  --c-data-type {HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT16,UINT16,SINT16,SINT16,SINT16,SINT16,UINT32,UINT32,SINT32,SINT32,SINT32,SINT32,UINT64,UINT64,SINT64,SINT64,SINT64,SINT64}
                        Type to use for A/B tiles
  --variant {0,1}       Variant to use. Details: 0: <A: SINGLE with 1vx1 B: SINGLE
                        with 1x1 C: SINGLE with 1vx1 > 1: <A: SINGLE with 1vx1 B:
                        SINGLE with 1vx1 C: SINGLE with 1vx1 >
  --m M                 m dimension in tiles
  --n N                 n dimension in tiles
  --k K                 k dimension in tiles
  --order ORDER         Loop order (default=mnkMNK)
  --column-strides {A,B,AB,C} [{A,B,AB,C} ...]
                        components with general column strides
  --row-strides {A,B,AB,C} [{A,B,AB,C} ...]
                        components with general row strides
  --A-data-regs A_DATA_REGS
                        Number of data registers to use for the A component
  --A-addr-regs A_ADDR_REGS
                        number of address registers to use for the A component
  --A-multiaddr-strat {interleave,split,phase}
                        Strategy for using multiple address registers for A
                        component
  --A-preload A_PRELOAD
                        number of A data registers to preload
  --B-data-regs B_DATA_REGS
                        Number of data registers to use for the B component
  --B-addr-regs B_ADDR_REGS
                        number of address registers to use for the B component
  --B-multiaddr-strat {interleave,split,phase}
                        Strategy for using multiple address registers for B
                        component
  --B-preload B_PRELOAD
                        number of B data registers to preload
  --AB-data-regs AB_DATA_REGS
                        Number of data registers to use for the AB component
  --AB-addr-regs AB_ADDR_REGS
                        number of address registers to use for the AB component
  --AB-multiaddr-strat {interleave,split,phase}
                        Strategy for using multiple address registers for AB
                        component
  --C-data-regs C_DATA_REGS
                        Number of data registers to use for the C component
  --C-addr-regs C_ADDR_REGS
                        number of address registers to use for the C component
  --C-multiaddr-strat {interleave,split,phase}
                        Strategy for using multiple address registers for C
                        component
  --sched-rar-distance SCHED_RAR_DISTANCE
                        Minimum read-after-read distance
  --sched-raw-distance SCHED_RAW_DISTANCE
                        Minimum read-after-write distance
  --sched-war-distance SCHED_WAR_DISTANCE
                        Minimum write-after-read distance
  --sched-waw-distance SCHED_WAW_DISTANCE
                        Minimum write-after-waw distance
  --output-filename OUTPUT_FILENAME
                        Path to the file to output the generated ASM function to
  --function-name FUNCTION_NAME
                        Override the name of the ASM symbol for the kernel function
```
(available arguments and argument choices change depending on other arguments)

## Example:

  - RVV
  - based on FMA vf-form instruction
  - dimensions: 2Vx12
  - unroll 8x
  - FP32 (sgemm)
  - use 4 vregs for A
  - use 8 fregs for B
  - preload all used regs
  - 2 address regs for each component
  - general column strides for C matrix
  - ensure 10 instructions of distance between raw dependencies
  - ensure 1 instructions of distance between war dependencies

```
$ python -m clitools.ukrgen --ukr gemm --isa rvv --op fma --ab-data-type SINGLE --c-data-type SINGLE --variant 0 --m 2 --n 12 --k 8 --column-strides C --A-data-regs 4 --B-data-regs 8 --C-data-regs 24 --AB-data-regs 24 --A-addr-regs 2 --B-addr-regs 2 --C-addr-regs 2 --B-multiaddr-strat interleave --A-preload 4 --B-preload 8 --sched-war-distance 1 --sched-raw-distance 10 --output-filename 2Vx12.s
[...]
DEBUG:GENERATOR:  RES:AB22                       : v26
DEBUG:GENERATOR:  RES:AB23                       : v27
DEBUG:GENERATOR:Aliased tregs in the end:
DEBUG:GENERATOR:Writing source to 2Vx12.s
```

## Legacy generator

see [legacy/OLDGENERATOR.md](legacy/OLDGENERATOR.md)
also see [asmgen-gemm](https://github.com/linedot/asmgen-gemm)


# License

ukrgen is distributed under the terms of both the MIT license and the GNU General Public License v3.0. Users may choose either license, at their option.

All new contributions must be made under both the MIT and GNU General Public License v3.0.

See LICENSE-GPL-3.0, LICENSE-MIT for details.

SPDX-License-Identifier: MIT OR GPL-3.0-or-later
