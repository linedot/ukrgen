# ukrgen - compute kernel generators based on asmgen

## New generator

WIP

Usage (MM kernel in stdout):
```
usage: ukrgen.py --isa {avx128,avx256,avx512,rvv,rvv071,neon,sve,sme} --op {fma,dota,fopa,mma}
                 --ab-data-type {HALF,HALF,HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT8,UINT8,SINT8,SINT8,SINT16,SINT16,UINT16,UINT16,SINT16,SINT16,SINT32,SINT32,UINT32,UINT32,SINT32,SINT32,SINT64,SINT64}
                 --c-data-type {HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT16,UINT16,SINT16,SINT16,SINT16,SINT16,UINT32,UINT32,SINT32,SINT32,SINT32,SINT32,UINT64,UINT64,SINT64,SINT64,SINT64,SINT64}
                 --variant {0,1} --m M --n N [--k K] [--order ORDER]
                 [--column-strides {a,b,c} [{a,b,c} ...]] [--row-strides {a,b,c} [{a,b,c} ...]]
                 --a-data-regs A_DATA_REGS --b-data-regs B_DATA_REGS --c-data-regs C_DATA_REGS
                 [--a-addr-regs A_ADDR_REGS] [--b-addr-regs B_ADDR_REGS] [--c-addr-regs C_ADDR_REGS]
                 --a-preload A_PRELOAD --b-preload B_PRELOAD
                 [--sched-rar-distance SCHED_RAR_DISTANCE] [--sched-raw-distance SCHED_RAW_DISTANCE]
                 [--sched-war-distance SCHED_WAR_DISTANCE] [--sched-waw-distance SCHED_WAW_DISTANCE]

ukrgen compute kernel generator

options:
  --isa {avx128,avx256,avx512,rvv,rvv071,neon,sve,sme}
                        ISA to use
  --op {fma,dota,fopa,mma}
                        Arithmetic operation
  --ab-data-type {HALF,HALF,HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT8,UINT8,SINT8,SINT8,SINT16,SINT16,UINT16,UINT16,SINT16,SINT16,SINT32,SINT32,UINT32,UINT32,SINT32,SINT32,SINT64,SINT64}
                        Type to use for A/B tiles
  --c-data-type {HALF,HALF,SINGLE,SINGLE,SINGLE,SINGLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SINT8,SINT8,UINT16,UINT16,SINT16,SINT16,SINT16,SINT16,UINT32,UINT32,SINT32,SINT32,SINT32,SINT32,UINT64,UINT64,SINT64,SINT64,SINT64,SINT64}
                        Type to use for A/B tiles
  --variant {0,1}       Variant to use
  --m M                 m dimension in tiles
  --n N                 n dimension in tiles
  --k K                 k dimension in tiles
  --order ORDER         Loop order (default=mnkMNK)
  --column-strides {a,b,c} [{a,b,c} ...]
                        components with general column strides
  --row-strides {a,b,c} [{a,b,c} ...]
                        components with general row strides
  --a-data-regs A_DATA_REGS
                        number of data registers to use for the a tiles
  --b-data-regs B_DATA_REGS
                        number of data registers to use for the b tiles
  --c-data-regs C_DATA_REGS
                        number of data registers to use for the c tiles
  --a-addr-regs A_ADDR_REGS
                        number of address registers to use for the a tiles
  --b-addr-regs B_ADDR_REGS
                        number of address registers to use for the b tiles
  --c-addr-regs C_ADDR_REGS
                        number of address registers to use for the c tiles
  --a-preload A_PRELOAD
                        number of a data registers to preload
  --b-preload B_PRELOAD
                        number of b data registers to preload
  --sched-rar-distance SCHED_RAR_DISTANCE
                        Minimum read-after-read distance
  --sched-raw-distance SCHED_RAW_DISTANCE
                        Minimum read-after-write distance
  --sched-war-distance SCHED_WAR_DISTANCE
                        Minimum write-after-read distance
  --sched-waw-distance SCHED_WAW_DISTANCE
                        Minimum write-after-waw distance

```
(available arguments and argument choices change depending on other arguments)

## Legacy generator

see [legacy/OLDGENERATOR.md](legacy/OLDGENERATOR.md)
also see [asmgen-gemm](https://github.com/linedot/asmgen-gemm)


# License

ukrgen is distributed under the terms of both the MIT license and the GNU General Public License v3.0. Users may choose either license, at their option.

All new contributions must be made under both the MIT and GNU General Public License v3.0.

See LICENSE-GPL-3.0, LICENSE-MIT for details.

SPDX-License-Identifier: MIT OR GPL-3.0-or-later
