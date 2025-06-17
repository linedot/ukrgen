# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest


from ukrgen.components import (
        dimension_properties,
        dimension_type,
        simple_ukr_tile,
        storage_type,
        tile,
        scalar_dp,
        scalar_tile,
        vla_vector,
        x4_vector
        )
from ukrgen.generators import mm,order2D
from ukrgen.models import load_store_cpu
from ukrgen.schedulers import simple_dependency_scheduler

# i.e SVE FP64 vector           is dima=(dt=vla,size=1,sdt=fixed,sd_size=2), dimb=(dt=fixed,size=1,sdt=fixed,sd_size=1)
#     SVE FP64 vector group(x4) is dima=(dt=vla,size=1,sdt=fixed,sd_size=2), dimb=(dt=fixed,size=4,sdt=fixed,sd_size=1)
#     (sd_size=1) for dimb because we can use the individual vectors
#     AVX FP64 vector           is dima=(dt=fixed,size=4,sdt=fixed,sd_size=4), dimb=(dt=fixed,size=1,sdt=fixed,sd_size=1)
#     RVV FP64 vector LMUL=4    is dima=(dt=vla,size=1,sdt=vla,sd_size=1), dimb=(dt=fixed,size=4,sdt=fixed,sd_size=1)
#     (sd_size=1) for dimb again because we can use the individual vectors (paying the cost for a vsetvl or using 
#      whole-vector instructions if available for the operation in question)
#     PTX m16n8k16 fragments:
#                             A is dima=(dt=fixed,size=16,std=fixed,fd_size=16), dimb=(dt=fixed,size=16,std=fixed,sd_size=16)
#                             B is dima=(dt=fixed,size=16,std=fixed,fd_size=16), dimb=(dt=fixed,size=8,std=fixed,sd_size=16)
#                             C is dima=(dt=fixed,size=16,std=fixed,fd_size=16), dimb=(dt=fixed,size=8,std=fixed,sd_size=16)
#     GPU stuff still needs some thinking because of the cooperative threads and complexity of indexing into the tiles

# Taxonomy of GEMM kernels:
# vlen = SIMD width in elements
# kernel dimensions : mxn
# mv = m/vlen
# nv = n/vlen
# kv = k/vlen
# VLA requirements:
#   - s: sliding scalar elements along vector registers
#   - l: loop over elements in vector registers
#   - rti: runtime index into vector
#   - rtr: Number of required registers to express kernel depends
#          on vlen (possible by implementing multiple kernels and 
#          selecting at runtime if vlen << N_{reg,arch})
#  Type             A            B             C          VLA  requirements   VLA dim
# fma_vf     | mv vectors  | n scalars   | mvxn vectors  |                  | m      |
# fma_idx    | mv vectors  | nv vectors  | mvxn vectors  | rtr(n),sl or rti | m      |
# fma_bcast  | mv vectors  | n vectors   | mvxn vectors  |                  | m      |
# fmat_vf    | m scalars   | nv vectors  | mxnv vectors  |                  | n      |
# fmat_idx   | mv vectors  | nv vectors  | mxnv vectors  | rtr(m),sl or rti | n      |
# fmat_bcast | m vectors   | nv vectors  | mxnv vectors  |                  | n      |
# dot_s      | m vectors   | n vectors   | mxn scalars   |                  | k      |
# dot_idx    | m vectors   | n vectors   | mvxn vectors  | rtr(m),sl or rti | k      |
# dott_idx   | m vectors   | n vectors   | mxnv vectors  | rtr(n),sl or rti | k      |
# opa        | mv vectors  | nv vectors  | mvxnv tiles   |                  | m,n    |
# mma        | mvxkv tiles | kvxnv tiles | mvxnv tiles   |                  | m,n,k  |
#
# With SVE we also have ability to lane-select within each fixed-size (128bit) chunk of
# the vector, enabling variants of fma_bcast that broadcast one chunk into the vector:
# nc = n/chunk size
#  Type             A            B             C          VLA  requirements   VLA dim
# fma_bcidx  | mv vectors  | nc vectors  | mvxn vectors  |                  | m      |
# fmat_bcidx | mc vectors  | nv vectors  | mxnv vectors  |                  | n      |
#
# - scalar fma is any fma but with vlen=1
# - number of operations is always cdims * k (or kv for mma)
# - The VLA-problematic ones are always when between a c dim and the corresponding A/B dim,
#   one is xv and the other is x

class test_mm(unittest.TestCase):
    # TODO: multitile
    #def test_multitile(self):
    #    m = [4,3]
    #    n = [4]
    #    k = [2]

    #    
    #    
    #    a_tile = composed_ukr_tile(a_sizes=m, b_sizes=k, subdims=(scalar_dp,scalar_dp))
    #    b_tile = composed_ukr_tile(a_sizes=k, b_sizes=n, subdims=(scalar_dp,scalar_dp))
    #    c_tile = composed_ukr_tile(a_sizes=m, b_sizes=n, subdims=(scalar_dp,scalar_dp))

    #    mmgen = mm(a_tile, b_tile, c_tile)
    #    inspector = string_mapper(op="fma")

    #    mm_ops = mmgen.generate()

    #    print("\n".join(map(str,inspector(mm_ops))))
    #    #self.assertEqual(expected_sequence, inspector(mm_ops))

    def test_2x4_scalar(self):
        m = 2
        n = 4
        unroll_factor = 1
        k = unroll_factor
        use_fma_vf = True

        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar_dp,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar_dp,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="fma")

        mm_ops = mmgen.generate()

        expected_sequence=[
            "C(0,0) <- fma(A(0,0),B(0,0),C(0,0))",
            "C(1,0) <- fma(A(1,0),B(0,0),C(1,0))",
            "C(0,1) <- fma(A(0,0),B(0,1),C(0,1))",
            "C(1,1) <- fma(A(1,0),B(0,1),C(1,1))",
            "C(0,2) <- fma(A(0,0),B(0,2),C(0,2))",
            "C(1,2) <- fma(A(1,0),B(0,2),C(1,2))",
            "C(0,3) <- fma(A(0,0),B(0,3),C(0,3))",
            "C(1,3) <- fma(A(1,0),B(0,3),C(1,3))"
        ]
        self.assertEqual(expected_sequence, list(map(str,mm_ops)))

        ac_mapper = lambda tile, idx : m*idx[1]+idx[0]
        b_mapper = lambda tile, idx : k*idx[0]+idx[1]
        machine = load_store_cpu(
                     res_counts=[2,4,8],
                     res_steps=[1,1,1],
                     addr_counts=[1,1,1],
                     addr_offset_ranges=[[(0,7)],[(0,7)],[(0,7)]],
                     addr_starts=[[0],[0],[0]],
                     preload_counts=[0,0,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper])

        expected_sequence = [
            "a0 <- LOAD aa0 + 0",
            "b0 <- LOAD ba0 + 0",
            "c0 <- fma(a0, b0, c0)",
            "a1 <- LOAD aa0 + 1",
            "c1 <- fma(a1, b0, c1)",
            "b1 <- LOAD ba0 + 1",
            "c2 <- fma(a0, b1, c2)",
            "c3 <- fma(a1, b1, c3)",
            "b2 <- LOAD ba0 + 2",
            "c4 <- fma(a0, b2, c4)",
            "c5 <- fma(a1, b2, c5)",
            "b3 <- LOAD ba0 + 3",
            "c6 <- fma(a0, b3, c6)",
            "c7 <- fma(a1, b3, c7)"
        ]
        self.assertEqual(expected_sequence, list(map(str,machine(mm_ops))))
        

    def test_2vx4_vla_vf_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="fma")


        mm_ops = mmgen.generate()

        expected_sequence = [
            "C(0*VLEN,0) <- fma(A(0*VLEN,0),B(0,0),C(0*VLEN,0))",
            "C(1*VLEN,0) <- fma(A(1*VLEN,0),B(0,0),C(1*VLEN,0))",
            "C(0*VLEN,1) <- fma(A(0*VLEN,0),B(0,1),C(0*VLEN,1))",
            "C(1*VLEN,1) <- fma(A(1*VLEN,0),B(0,1),C(1*VLEN,1))",
            "C(0*VLEN,2) <- fma(A(0*VLEN,0),B(0,2),C(0*VLEN,2))",
            "C(1*VLEN,2) <- fma(A(1*VLEN,0),B(0,2),C(1*VLEN,2))",
            "C(0*VLEN,3) <- fma(A(0*VLEN,0),B(0,3),C(0*VLEN,3))",
            "C(1*VLEN,3) <- fma(A(1*VLEN,0),B(0,3),C(1*VLEN,3))",
            "C(0*VLEN,0) <- fma(A(0*VLEN,1),B(1,0),C(0*VLEN,0))",
            "C(1*VLEN,0) <- fma(A(1*VLEN,1),B(1,0),C(1*VLEN,0))",
            "C(0*VLEN,1) <- fma(A(0*VLEN,1),B(1,1),C(0*VLEN,1))",
            "C(1*VLEN,1) <- fma(A(1*VLEN,1),B(1,1),C(1*VLEN,1))",
            "C(0*VLEN,2) <- fma(A(0*VLEN,1),B(1,2),C(0*VLEN,2))",
            "C(1*VLEN,2) <- fma(A(1*VLEN,1),B(1,2),C(1*VLEN,2))",
            "C(0*VLEN,3) <- fma(A(0*VLEN,1),B(1,3),C(0*VLEN,3))",
            "C(1*VLEN,3) <- fma(A(1*VLEN,1),B(1,3),C(1*VLEN,3))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mm_ops)))

    
        scale_tile = simple_ukr_tile(a_size=m*n, b_size=1, subdims=(vla_vector,scalar_dp))
        beta_tile = simple_ukr_tile(a_size=1, b_size=1, subdims=(scalar_dp,scalar_dp))
        scalegen = mm(scale_tile, beta_tile, scale_tile, opstr="mulnop", tile_strs=["C","beta","C"])

        beta_scale_ops = scalegen.generate()

        expected_sequence = [
            "C(0*VLEN,0) <- mulnop(C(0*VLEN,0),beta(0,0),C(0*VLEN,0))",
            "C(1*VLEN,0) <- mulnop(C(1*VLEN,0),beta(0,0),C(1*VLEN,0))",
            "C(2*VLEN,0) <- mulnop(C(2*VLEN,0),beta(0,0),C(2*VLEN,0))",
            "C(3*VLEN,0) <- mulnop(C(3*VLEN,0),beta(0,0),C(3*VLEN,0))",
            "C(4*VLEN,0) <- mulnop(C(4*VLEN,0),beta(0,0),C(4*VLEN,0))",
            "C(5*VLEN,0) <- mulnop(C(5*VLEN,0),beta(0,0),C(5*VLEN,0))",
            "C(6*VLEN,0) <- mulnop(C(6*VLEN,0),beta(0,0),C(6*VLEN,0))",
            "C(7*VLEN,0) <- mulnop(C(7*VLEN,0),beta(0,0),C(7*VLEN,0))"
        ]
        self.assertEqual(expected_sequence, list(map(str,beta_scale_ops)))

        scalegen = mm(scale_tile, beta_tile, scale_tile, opstr="fma", tile_strs=["AB","alpha","C"])
        alpha_scale_ops = scalegen.generate()

        expected_sequence = [
            "C(0*VLEN,0) <- fma(AB(0*VLEN,0),alpha(0,0),C(0*VLEN,0))",
            "C(1*VLEN,0) <- fma(AB(1*VLEN,0),alpha(0,0),C(1*VLEN,0))",
            "C(2*VLEN,0) <- fma(AB(2*VLEN,0),alpha(0,0),C(2*VLEN,0))",
            "C(3*VLEN,0) <- fma(AB(3*VLEN,0),alpha(0,0),C(3*VLEN,0))",
            "C(4*VLEN,0) <- fma(AB(4*VLEN,0),alpha(0,0),C(4*VLEN,0))",
            "C(5*VLEN,0) <- fma(AB(5*VLEN,0),alpha(0,0),C(5*VLEN,0))",
            "C(6*VLEN,0) <- fma(AB(6*VLEN,0),alpha(0,0),C(6*VLEN,0))",
            "C(7*VLEN,0) <- fma(AB(7*VLEN,0),alpha(0,0),C(7*VLEN,0))"
        ]
        self.assertEqual(expected_sequence, list(map(str,alpha_scale_ops)))

        ac_mapper = lambda tile, idx : m*idx[1]+idx[0]
        b_mapper = lambda tile, idx : n*idx[0]+idx[1]
        

        machine = load_store_cpu(
                     res_counts=[4,4,8],
                     res_steps=[1,1,1],
                     addr_counts=[2,1,1],
                     addr_offset_ranges=[[(0,0),(0,0)],[(0,7)],[(0,0)]],
                     addr_starts=[[0,1],[0],[0]],
                     preload_counts=[2,3,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper])

        mm_ops_next = mmgen.generate(add_dims=[0,0,0,0,0,k])
        #print("\n".join(map(str,inspector(mm_ops_next))))

        preload = machine.preload(mm_ops)
        mainblock = machine(mm_ops)
        storeblock = machine.store_modified()
        preload_mb = machine.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])

        expected_preload = [
            "a0 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 0",
            "c0 <- 0",
            "a1 <- LOAD aa1 + 0*VLEN",
            "c1 <- 0",
            "b1 <- LOAD ba0 + 1",
            "c2 <- 0",
            "c3 <- 0",
            "b2 <- LOAD ba0 + 2",
            "c4 <- 0",
            "c5 <- 0",
            "c6 <- 0",
            "c7 <- 0",
            "aa0 <- aa0 + 2*VLEN",
            "aa1 <- aa1 + 2*VLEN",
            "ba0 <- ba0 + 3",
        ]
        expected_mainblock = [
            "c0 <- fma(a0, b0, c0)",
            "c1 <- fma(a1, b0, c1)",
            "c2 <- fma(a0, b1, c2)",
            "c3 <- fma(a1, b1, c3)",
            "c4 <- fma(a0, b2, c4)",
            "c5 <- fma(a1, b2, c5)",
            "b3 <- LOAD ba0 + 0",
            "c6 <- fma(a0, b3, c6)",
            "c7 <- fma(a1, b3, c7)",
            "a2 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 1",
            "c0 <- fma(a2, b0, c0)",
            "a3 <- LOAD aa1 + 0*VLEN",
            "c1 <- fma(a3, b0, c1)",
            "b1 <- LOAD ba0 + 2",
            "c2 <- fma(a2, b1, c2)",
            "c3 <- fma(a3, b1, c3)",
            "b2 <- LOAD ba0 + 3",
            "c4 <- fma(a2, b2, c4)",
            "c5 <- fma(a3, b2, c5)",
            "b3 <- LOAD ba0 + 4",
            "c6 <- fma(a2, b3, c6)",
            "c7 <- fma(a3, b3, c7)",
        ]
        expected_preload_mb = [
            "aa0 <- aa0 + 2*VLEN",
            "a0 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 5",
            "aa1 <- aa1 + 2*VLEN",
            "a1 <- LOAD aa1 + 0*VLEN",
            "b1 <- LOAD ba0 + 6",
            "b2 <- LOAD ba0 + 7",
            "aa0 <- aa0 + 2*VLEN",
            "aa1 <- aa1 + 2*VLEN",
            "ba0 <- ba0 + 8",
        ]
        expected_storeblock = [
            "ca0 + 0*VLEN <- STORE c0",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c1",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c2",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c3",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c4",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c5",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c6",
            "ca0 <- ca0 + 1*VLEN",
            "ca0 + 0*VLEN <- STORE c7",
        ]
        
        #print("\n".join(map(str,preload)))
        #print("MAIN LOOP -------------------------------")
        #print("  "+"\n  ".join(map(str,mainblock)))
        #print("PRELOAD NEXT ----------------------------")
        #print("  "+"\n  ".join(map(str,preload_mb)))
        #print("END MAIN LOOP ---------------------------")
        #print("STOREBLOCK ------------------------------")
        #print("  "+"\n  ".join(map(str,storeblock)))
        #print("ENDSTOREBLOCK ---------------------------")

        self.assertEqual(expected_preload,    list(map(str,preload)))
        self.assertEqual(expected_mainblock,  list(map(str,mainblock)))
        self.assertEqual(expected_storeblock,  list(map(str,storeblock)))
        self.assertEqual(expected_preload_mb, list(map(str,preload_mb)))

        scheduler = simple_dependency_scheduler(rar=0,raw=6,war=0,waw=6)

        rs_preload = scheduler(preload, loop=False)
        rs_mbpl = scheduler(mainblock+preload_mb)

        expected_rs_preload = list(map(str,preload))

        expected_rs_mbpl = [
            "b3 <- LOAD ba0 + 0",
            "c0 <- fma(a0, b0, c0)",
            "c1 <- fma(a1, b0, c1)",
            "a2 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 1",
            "c2 <- fma(a0, b1, c2)",
            "a3 <- LOAD aa1 + 0*VLEN",
            "c3 <- fma(a1, b1, c3)",
            "b1 <- LOAD ba0 + 2",
            "c4 <- fma(a0, b2, c4)",
            "c5 <- fma(a1, b2, c5)",
            "b2 <- LOAD ba0 + 3",
            "c6 <- fma(a0, b3, c6)",
            "c7 <- fma(a1, b3, c7)",
            "b3 <- LOAD ba0 + 4",
            "c0 <- fma(a2, b0, c0)",
            "c1 <- fma(a3, b0, c1)",
            "aa0 <- aa0 + 2*VLEN",
            "c2 <- fma(a2, b1, c2)",
            "c3 <- fma(a3, b1, c3)",
            "aa1 <- aa1 + 2*VLEN",
            "c4 <- fma(a2, b2, c4)",
            "c5 <- fma(a3, b2, c5)",
            "b0 <- LOAD ba0 + 5",
            "b1 <- LOAD ba0 + 6",
            "b2 <- LOAD ba0 + 7",
            "ba0 <- ba0 + 8",
            "a0 <- LOAD aa0 + 0*VLEN",
            "a1 <- LOAD aa1 + 0*VLEN",
            "aa0 <- aa0 + 2*VLEN",
            "c6 <- fma(a2, b3, c6)",
            "c7 <- fma(a3, b3, c7)",
            "aa1 <- aa1 + 2*VLEN",
        ]

        self.assertEqual(expected_rs_preload,    list(map(str,rs_preload)))
        self.assertEqual(expected_rs_mbpl,  list(map(str,rs_mbpl)))

    def test_2vx4_fix_vf_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(x4_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(x4_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="fma")

        mm_ops = mmgen.generate()
        expected_sequence = [
            "C(0,0) <- fma(A(0,0),B(0,0),C(0,0))",
            "C(4,0) <- fma(A(4,0),B(0,0),C(4,0))",
            "C(0,1) <- fma(A(0,0),B(0,1),C(0,1))",
            "C(4,1) <- fma(A(4,0),B(0,1),C(4,1))",
            "C(0,2) <- fma(A(0,0),B(0,2),C(0,2))",
            "C(4,2) <- fma(A(4,0),B(0,2),C(4,2))",
            "C(0,3) <- fma(A(0,0),B(0,3),C(0,3))",
            "C(4,3) <- fma(A(4,0),B(0,3),C(4,3))",
            "C(0,0) <- fma(A(0,1),B(1,0),C(0,0))",
            "C(4,0) <- fma(A(4,1),B(1,0),C(4,0))",
            "C(0,1) <- fma(A(0,1),B(1,1),C(0,1))",
            "C(4,1) <- fma(A(4,1),B(1,1),C(4,1))",
            "C(0,2) <- fma(A(0,1),B(1,2),C(0,2))",
            "C(4,2) <- fma(A(4,1),B(1,2),C(4,2))",
            "C(0,3) <- fma(A(0,1),B(1,3),C(0,3))",
            "C(4,3) <- fma(A(4,1),B(1,3),C(4,3))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mm_ops)))

    # TODO: more metainfo required to differentiate fma_bcastx4 from this
    def test_2vx4_idxx4_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(x4_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n//x4_vector.sd_size, subdims=(scalar_dp,x4_vector))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(x4_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, lo=order2D('mkMnNK'),opstr="fma_idxx4")

        mm_ops = mmgen.generate()

        expected_sequence = [
            "C(0,0+0) <- fma_idxx4(A(0,0),B(0,0.el[0]),C(0,0+0))",
            "C(4,0+0) <- fma_idxx4(A(4,0),B(0,0.el[0]),C(4,0+0))",
            "C(0,0+1) <- fma_idxx4(A(0,0),B(0,0.el[1]),C(0,0+1))",
            "C(4,0+1) <- fma_idxx4(A(4,0),B(0,0.el[1]),C(4,0+1))",
            "C(0,0+2) <- fma_idxx4(A(0,0),B(0,0.el[2]),C(0,0+2))",
            "C(4,0+2) <- fma_idxx4(A(4,0),B(0,0.el[2]),C(4,0+2))",
            "C(0,0+3) <- fma_idxx4(A(0,0),B(0,0.el[3]),C(0,0+3))",
            "C(4,0+3) <- fma_idxx4(A(4,0),B(0,0.el[3]),C(4,0+3))",
            "C(0,0+0) <- fma_idxx4(A(0,1),B(1,0.el[0]),C(0,0+0))",
            "C(4,0+0) <- fma_idxx4(A(4,1),B(1,0.el[0]),C(4,0+0))",
            "C(0,0+1) <- fma_idxx4(A(0,1),B(1,0.el[1]),C(0,0+1))",
            "C(4,0+1) <- fma_idxx4(A(4,1),B(1,0.el[1]),C(4,0+1))",
            "C(0,0+2) <- fma_idxx4(A(0,1),B(1,0.el[2]),C(0,0+2))",
            "C(4,0+2) <- fma_idxx4(A(4,1),B(1,0.el[2]),C(4,0+2))",
            "C(0,0+3) <- fma_idxx4(A(0,1),B(1,0.el[3]),C(0,0+3))",
            "C(4,0+3) <- fma_idxx4(A(4,1),B(1,0.el[3]),C(4,0+3))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mm_ops)))

        ac_mapper = lambda tile, idx : tile.dima.size*m*idx[1]+idx[0]
        b_mapper = lambda tile, idx : n*idx[0]+idx[1]
        

        machine = load_store_cpu(
                     res_counts=[4,2,8],
                     res_steps=[1,1,1],
                     addr_counts=[2,1,1],
                     addr_offset_ranges=[[(0,255),(0,255)],[(0,7)],[(0,255)]],
                     addr_starts=[[0,4],[0],[0]],
                     preload_counts=[2,2,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper])

        mm_ops_next = mmgen.generate(add_dims=[0,0,0,0,0,k])
        #print("\n".join(map(str,inspector(mm_ops_next))))

        preload = machine.preload(mm_ops)
        mainblock = machine(mm_ops)
        storeblock = machine.store_modified()
        preload_mb = machine.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])

        #print("\n".join(map(str,preload)))
        #print("MAIN LOOP -------------------------------")
        #print("  "+"\n  ".join(map(str,mainblock)))
        #print("PRELOAD NEXT ----------------------------")
        #print("  "+"\n  ".join(map(str,preload_mb)))
        #print("END MAIN LOOP ---------------------------")
        #print("STOREBLOCK ------------------------------")
        #print("  "+"\n  ".join(map(str,storeblock)))
        #print("ENDSTOREBLOCK ---------------------------")

        expected_preload = [
            "a0 <- LOAD aa0 + 0",
            "b0 <- LOAD ba0 + 0",
            "c0 <- 0",
            "a1 <- LOAD aa1 + 0",
            "c1 <- 0",
            "c2 <- 0",
            "c3 <- 0",
            "c4 <- 0",
            "c5 <- 0",
            "c6 <- 0",
            "c7 <- 0",
            "b1 <- LOAD ba0 + 4",
            "aa0 <- aa0 + 8",
            "aa1 <- aa1 + 8",
            "ba0 <- ba0 + 8",
        ]
        expected_mainblock = [
            "c0 <- fma(a0, b0.el[0], c0)",
            "c1 <- fma(a1, b0.el[0], c1)",
            "c2 <- fma(a0, b0.el[1], c2)",
            "c3 <- fma(a1, b0.el[1], c3)",
            "c4 <- fma(a0, b0.el[2], c4)",
            "c5 <- fma(a1, b0.el[2], c5)",
            "c6 <- fma(a0, b0.el[3], c6)",
            "c7 <- fma(a1, b0.el[3], c7)",
            "a2 <- LOAD aa0 + 0",
            "c0 <- fma(a2, b1.el[0], c0)",
            "a3 <- LOAD aa1 + 0",
            "c1 <- fma(a3, b1.el[0], c1)",
            "c2 <- fma(a2, b1.el[1], c2)",
            "c3 <- fma(a3, b1.el[1], c3)",
            "c4 <- fma(a2, b1.el[2], c4)",
            "c5 <- fma(a3, b1.el[2], c5)",
            "c6 <- fma(a2, b1.el[3], c6)",
            "c7 <- fma(a3, b1.el[3], c7)",
        ]
        expected_preload_mb = [
            "a0 <- LOAD aa1 + 4", # TODO: aa0 + 8
            "b0 <- LOAD ba0 + 0",
            "a1 <- LOAD aa1 + 8",
            "b1 <- LOAD ba0 + 4",
            "aa0 <- aa0 + 16",
            "aa1 <- aa1 + 16",
            "ba0 <- ba0 + 8",
        ]
        expected_storeblock = [
            "ca0 + 0 <- STORE c0",
            "ca0 + 4 <- STORE c1",
            "ca0 + 8 <- STORE c2",
            "ca0 + 12 <- STORE c3",
            "ca0 + 16 <- STORE c4",
            "ca0 + 20 <- STORE c5",
            "ca0 + 24 <- STORE c6",
            "ca0 + 28 <- STORE c7",
        ]

        self.assertEqual(expected_preload,    list(map(str,preload)))
        self.assertEqual(expected_mainblock,  list(map(str,mainblock)))
        self.assertEqual(expected_storeblock,  list(map(str,storeblock)))
        self.assertEqual(expected_preload_mb, list(map(str,preload_mb)))

    def test_2vx4_fix_dot(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar_dp,x4_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(x4_vector,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar_dp,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="dota")

        expected_sequence = [
            "C(0,0) <- dota(A(0,0),B(0,0),C(0,0))",
            "C(1,0) <- dota(A(1,0),B(0,0),C(1,0))",
            "C(0,1) <- dota(A(0,0),B(0,1),C(0,1))",
            "C(1,1) <- dota(A(1,0),B(0,1),C(1,1))",
            "C(0,2) <- dota(A(0,0),B(0,2),C(0,2))",
            "C(1,2) <- dota(A(1,0),B(0,2),C(1,2))",
            "C(0,3) <- dota(A(0,0),B(0,3),C(0,3))",
            "C(1,3) <- dota(A(1,0),B(0,3),C(1,3))",
            "C(0,0) <- dota(A(0,4),B(4,0),C(0,0))",
            "C(1,0) <- dota(A(1,4),B(4,0),C(1,0))",
            "C(0,1) <- dota(A(0,4),B(4,1),C(0,1))",
            "C(1,1) <- dota(A(1,4),B(4,1),C(1,1))",
            "C(0,2) <- dota(A(0,4),B(4,2),C(0,2))",
            "C(1,2) <- dota(A(1,4),B(4,2),C(1,2))",
            "C(0,3) <- dota(A(0,4),B(4,3),C(0,3))",
            "C(1,3) <- dota(A(1,4),B(4,3),C(1,3))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mmgen.generate())))

    def test_2vx4_vla_dot(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar_dp,vla_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(vla_vector,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar_dp,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="dota")

        expected_sequence = [
            "C(0,0) <- dota(A(0,0*VLEN),B(0*VLEN,0),C(0,0))",
            "C(1,0) <- dota(A(1,0*VLEN),B(0*VLEN,0),C(1,0))",
            "C(0,1) <- dota(A(0,0*VLEN),B(0*VLEN,1),C(0,1))",
            "C(1,1) <- dota(A(1,0*VLEN),B(0*VLEN,1),C(1,1))",
            "C(0,2) <- dota(A(0,0*VLEN),B(0*VLEN,2),C(0,2))",
            "C(1,2) <- dota(A(1,0*VLEN),B(0*VLEN,2),C(1,2))",
            "C(0,3) <- dota(A(0,0*VLEN),B(0*VLEN,3),C(0,3))",
            "C(1,3) <- dota(A(1,0*VLEN),B(0*VLEN,3),C(1,3))",
            "C(0,0) <- dota(A(0,1*VLEN),B(1*VLEN,0),C(0,0))",
            "C(1,0) <- dota(A(1,1*VLEN),B(1*VLEN,0),C(1,0))",
            "C(0,1) <- dota(A(0,1*VLEN),B(1*VLEN,1),C(0,1))",
            "C(1,1) <- dota(A(1,1*VLEN),B(1*VLEN,1),C(1,1))",
            "C(0,2) <- dota(A(0,1*VLEN),B(1*VLEN,2),C(0,2))",
            "C(1,2) <- dota(A(1,1*VLEN),B(1*VLEN,2),C(1,2))",
            "C(0,3) <- dota(A(0,1*VLEN),B(1*VLEN,3),C(0,3))",
            "C(1,3) <- dota(A(1,1*VLEN),B(1*VLEN,3),C(1,3))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mmgen.generate())))

    def test_4vx4_idxx4_dot(self):
        m = 4
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar_dp,x4_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(x4_vector,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m//x4_vector.sd_size, b_size=n, subdims=(x4_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="dota")

        expected_sequence = [
            "C(0.el[0],0) <- dota(A(0+0,0),B(0,0),C(0.el[0],0))",
            "C(0.el[1],0) <- dota(A(0+1,0),B(0,0),C(0.el[1],0))",
            "C(0.el[2],0) <- dota(A(0+2,0),B(0,0),C(0.el[2],0))",
            "C(0.el[3],0) <- dota(A(0+3,0),B(0,0),C(0.el[3],0))",
            "C(0.el[0],1) <- dota(A(0+0,0),B(0,1),C(0.el[0],1))",
            "C(0.el[1],1) <- dota(A(0+1,0),B(0,1),C(0.el[1],1))",
            "C(0.el[2],1) <- dota(A(0+2,0),B(0,1),C(0.el[2],1))",
            "C(0.el[3],1) <- dota(A(0+3,0),B(0,1),C(0.el[3],1))",
            "C(0.el[0],2) <- dota(A(0+0,0),B(0,2),C(0.el[0],2))",
            "C(0.el[1],2) <- dota(A(0+1,0),B(0,2),C(0.el[1],2))",
            "C(0.el[2],2) <- dota(A(0+2,0),B(0,2),C(0.el[2],2))",
            "C(0.el[3],2) <- dota(A(0+3,0),B(0,2),C(0.el[3],2))",
            "C(0.el[0],3) <- dota(A(0+0,0),B(0,3),C(0.el[0],3))",
            "C(0.el[1],3) <- dota(A(0+1,0),B(0,3),C(0.el[1],3))",
            "C(0.el[2],3) <- dota(A(0+2,0),B(0,3),C(0.el[2],3))",
            "C(0.el[3],3) <- dota(A(0+3,0),B(0,3),C(0.el[3],3))",
            "C(0.el[0],0) <- dota(A(0+0,4),B(4,0),C(0.el[0],0))",
            "C(0.el[1],0) <- dota(A(0+1,4),B(4,0),C(0.el[1],0))",
            "C(0.el[2],0) <- dota(A(0+2,4),B(4,0),C(0.el[2],0))",
            "C(0.el[3],0) <- dota(A(0+3,4),B(4,0),C(0.el[3],0))",
            "C(0.el[0],1) <- dota(A(0+0,4),B(4,1),C(0.el[0],1))",
            "C(0.el[1],1) <- dota(A(0+1,4),B(4,1),C(0.el[1],1))",
            "C(0.el[2],1) <- dota(A(0+2,4),B(4,1),C(0.el[2],1))",
            "C(0.el[3],1) <- dota(A(0+3,4),B(4,1),C(0.el[3],1))",
            "C(0.el[0],2) <- dota(A(0+0,4),B(4,2),C(0.el[0],2))",
            "C(0.el[1],2) <- dota(A(0+1,4),B(4,2),C(0.el[1],2))",
            "C(0.el[2],2) <- dota(A(0+2,4),B(4,2),C(0.el[2],2))",
            "C(0.el[3],2) <- dota(A(0+3,4),B(4,2),C(0.el[3],2))",
            "C(0.el[0],3) <- dota(A(0+0,4),B(4,3),C(0.el[0],3))",
            "C(0.el[1],3) <- dota(A(0+1,4),B(4,3),C(0.el[1],3))",
            "C(0.el[2],3) <- dota(A(0+2,4),B(4,3),C(0.el[2],3))",
            "C(0.el[3],3) <- dota(A(0+3,4),B(4,3),C(0.el[3],3))",
        ]
        
        #print("\n".join(map(str,preload)))
        #print("MAIN LOOP -------------------------------")
        #print("  "+"\n  ".join(map(str,mainblock)))
        #print("PRELOAD NEXT ----------------------------")
        #print("  "+"\n  ".join(map(str,preload_mb)))
        #print("END MAIN LOOP ---------------------------")
        #print("STOREBLOCK ------------------------------")
        #print("  "+"\n  ".join(map(str,storeblock)))
        #print("ENDSTOREBLOCK ---------------------------")
        self.assertEqual(expected_sequence, list(map(str,mmgen.generate())))

    def test_2vx4_vector_opa(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,vla_vector))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,vla_vector))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="opa")

        mm_ops = mmgen.generate()

        expected_sequence=[
            "C(0*VLEN,0*VLEN) <- opa(A(0*VLEN,0),B(0,0*VLEN),C(0*VLEN,0*VLEN))",
            "C(1*VLEN,0*VLEN) <- opa(A(1*VLEN,0),B(0,0*VLEN),C(1*VLEN,0*VLEN))",
            "C(0*VLEN,1*VLEN) <- opa(A(0*VLEN,0),B(0,1*VLEN),C(0*VLEN,1*VLEN))",
            "C(1*VLEN,1*VLEN) <- opa(A(1*VLEN,0),B(0,1*VLEN),C(1*VLEN,1*VLEN))",
            "C(0*VLEN,2*VLEN) <- opa(A(0*VLEN,0),B(0,2*VLEN),C(0*VLEN,2*VLEN))",
            "C(1*VLEN,2*VLEN) <- opa(A(1*VLEN,0),B(0,2*VLEN),C(1*VLEN,2*VLEN))",
            "C(0*VLEN,3*VLEN) <- opa(A(0*VLEN,0),B(0,3*VLEN),C(0*VLEN,3*VLEN))",
            "C(1*VLEN,3*VLEN) <- opa(A(1*VLEN,0),B(0,3*VLEN),C(1*VLEN,3*VLEN))",
            "C(0*VLEN,0*VLEN) <- opa(A(0*VLEN,1),B(1,0*VLEN),C(0*VLEN,0*VLEN))",
            "C(1*VLEN,0*VLEN) <- opa(A(1*VLEN,1),B(1,0*VLEN),C(1*VLEN,0*VLEN))",
            "C(0*VLEN,1*VLEN) <- opa(A(0*VLEN,1),B(1,1*VLEN),C(0*VLEN,1*VLEN))",
            "C(1*VLEN,1*VLEN) <- opa(A(1*VLEN,1),B(1,1*VLEN),C(1*VLEN,1*VLEN))",
            "C(0*VLEN,2*VLEN) <- opa(A(0*VLEN,1),B(1,2*VLEN),C(0*VLEN,2*VLEN))",
            "C(1*VLEN,2*VLEN) <- opa(A(1*VLEN,1),B(1,2*VLEN),C(1*VLEN,2*VLEN))",
            "C(0*VLEN,3*VLEN) <- opa(A(0*VLEN,1),B(1,3*VLEN),C(0*VLEN,3*VLEN))",
            "C(1*VLEN,3*VLEN) <- opa(A(1*VLEN,1),B(1,3*VLEN),C(1*VLEN,3*VLEN))",
        ]
        self.assertEqual(expected_sequence, list(map(str,mm_ops)))

        ac_mapper = lambda tile, idx : m*idx[1]+idx[0]
        b_mapper = lambda tile, idx : n*idx[0]+idx[1]
        

        machine = load_store_cpu(
                     res_counts=[4,8,8],
                     res_steps=[1,1,1],
                     addr_counts=[1,1,1],
                     addr_offset_ranges=[
                         [(0,7)],
                         [(0,7)],
                         [(0,0)]],
                     addr_starts=[
                         [0],
                         [0],
                         [0]],
                     preload_counts=[4,4,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper],
                     op="opa")

        mm_ops_next = mmgen.generate(add_dims=[0,0,0,0,0,k])
        #print("\n".join(map(str,inspector(mm_ops_next))))

        preload = machine.preload(mm_ops)
        mainblock = machine(mm_ops)
        preload_mb = machine.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])
        
        #print("\n".join(map(str,preload)))
        #print("MAIN LOOP -------------------------------")
        #print("  "+"\n  ".join(map(str,mainblock)))
        #print("PRELOAD NEXT ----------------------------")
        #print("  "+"\n  ".join(map(str,preload_mb)))
        #print("END MAIN LOOP ---------------------------")

        expected_preload = [
            "a0 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 0*VLEN",
            "c0 <- 0",
            "a1 <- LOAD aa0 + 1*VLEN",
            "c1 <- 0",
            "b1 <- LOAD ba0 + 1*VLEN",
            "c2 <- 0",
            "c3 <- 0",
            "b2 <- LOAD ba0 + 2*VLEN",
            "c4 <- 0",
            "c5 <- 0",
            "b3 <- LOAD ba0 + 3*VLEN",
            "c6 <- 0",
            "c7 <- 0",
            "a2 <- LOAD aa0 + 2*VLEN",
            "a3 <- LOAD aa0 + 3*VLEN",
            "aa0 <- aa0 + 4*VLEN",
            "ba0 <- ba0 + 4*VLEN",
        ]
        expected_mainblock = [
            "c0 <- opa(a0, b0, c0)",
            "c1 <- opa(a1, b0, c1)",
            "c2 <- opa(a0, b1, c2)",
            "c3 <- opa(a1, b1, c3)",
            "c4 <- opa(a0, b2, c4)",
            "c5 <- opa(a1, b2, c5)",
            "c6 <- opa(a0, b3, c6)",
            "c7 <- opa(a1, b3, c7)",
            "b4 <- LOAD ba0 + 0*VLEN",
            "c0 <- opa(a2, b4, c0)",
            "c1 <- opa(a3, b4, c1)",
            "b5 <- LOAD ba0 + 1*VLEN",
            "c2 <- opa(a2, b5, c2)",
            "c3 <- opa(a3, b5, c3)",
            "b6 <- LOAD ba0 + 2*VLEN",
            "c4 <- opa(a2, b6, c4)",
            "c5 <- opa(a3, b6, c5)",
            "b7 <- LOAD ba0 + 3*VLEN",
            "c6 <- opa(a2, b7, c6)",
            "c7 <- opa(a3, b7, c7)",
        ]

        expected_preload_mb = [
            "a0 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 4*VLEN",
            "a1 <- LOAD aa0 + 1*VLEN",
            "b1 <- LOAD ba0 + 5*VLEN",
            "b2 <- LOAD ba0 + 6*VLEN",
            "b3 <- LOAD ba0 + 7*VLEN",
            "a2 <- LOAD aa0 + 2*VLEN",
            "a3 <- LOAD aa0 + 3*VLEN",
            "aa0 <- aa0 + 4*VLEN",
            "ba0 <- ba0 + 8*VLEN",
        ]

        self.assertEqual(expected_preload,    list(map(str,preload)))
        self.assertEqual(expected_mainblock,  list(map(str,mainblock)))
        self.assertEqual(expected_preload_mb, list(map(str,preload_mb)))

        scheduler = simple_dependency_scheduler(rar=0,raw=4,war=0,waw=2)

        rs_preload = scheduler(preload, loop=False)
        rs_mbpl = scheduler(mainblock+preload_mb)

        #print("============== RESCHEDULED =================")
        #print("\n".join(map(str,rs_preload)))
        #print("MAIN LOOP -------------------------------")
        #print("  "+"\n  ".join(map(str,rs_mbpl)))
        #print("END MAIN LOOP ---------------------------")

        expected_rs_preload = [
            "a0 <- LOAD aa0 + 0*VLEN",
            "b0 <- LOAD ba0 + 0*VLEN",
            "c0 <- 0",
            "a1 <- LOAD aa0 + 1*VLEN",
            "c1 <- 0",
            "b1 <- LOAD ba0 + 1*VLEN",
            "c2 <- 0",
            "c3 <- 0",
            "b2 <- LOAD ba0 + 2*VLEN",
            "c4 <- 0",
            "c5 <- 0",
            "b3 <- LOAD ba0 + 3*VLEN",
            "c6 <- 0",
            "c7 <- 0",
            "a2 <- LOAD aa0 + 2*VLEN",
            "a3 <- LOAD aa0 + 3*VLEN",
            "aa0 <- aa0 + 4*VLEN",
            "ba0 <- ba0 + 4*VLEN",
        ]

        expected_rs_mbpl = [
            "b4 <- LOAD ba0 + 0*VLEN",
            "c1 <- opa(a1, b0, c1)",
            "c0 <- opa(a0, b0, c0)",
            "b5 <- LOAD ba0 + 1*VLEN",
            "c3 <- opa(a1, b1, c3)",
            "c2 <- opa(a0, b1, c2)",
            "b6 <- LOAD ba0 + 2*VLEN",
            "c5 <- opa(a1, b2, c5)",
            "c4 <- opa(a0, b2, c4)",
            "b7 <- LOAD ba0 + 3*VLEN",
            "c7 <- opa(a1, b3, c7)",
            "c6 <- opa(a0, b3, c6)",
            "c0 <- opa(a2, b4, c0)",
            "c1 <- opa(a3, b4, c1)",
            "c2 <- opa(a2, b5, c2)",
            "c3 <- opa(a3, b5, c3)",
            "c4 <- opa(a2, b6, c4)",
            "b0 <- LOAD ba0 + 4*VLEN",
            "b1 <- LOAD ba0 + 5*VLEN",
            "b2 <- LOAD ba0 + 6*VLEN",
            "b3 <- LOAD ba0 + 7*VLEN",
            "ba0 <- ba0 + 8*VLEN",
            "a1 <- LOAD aa0 + 1*VLEN",
            "a0 <- LOAD aa0 + 0*VLEN",
            "c5 <- opa(a3, b6, c5)",
            "c6 <- opa(a2, b7, c6)",
            "c7 <- opa(a3, b7, c7)",
            "a2 <- LOAD aa0 + 2*VLEN",
            "a3 <- LOAD aa0 + 3*VLEN",
            "aa0 <- aa0 + 4*VLEN",
        ]

       
        self.assertEqual(expected_rs_preload,    list(map(str,rs_preload)))
        self.assertEqual(expected_rs_mbpl,  list(map(str,rs_mbpl)))


    def test_2x2x2_tile_mma(self):
        m = 2
        n = 2
        k = 2
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(x4_vector,x4_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(x4_vector,x4_vector))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(x4_vector,x4_vector))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="mma")

        expected_sequence=[
            "C(0,0) <- mma(A(0,0),B(0,0),C(0,0))",
            "C(4,0) <- mma(A(4,0),B(0,0),C(4,0))",
            "C(0,4) <- mma(A(0,0),B(0,4),C(0,4))",
            "C(4,4) <- mma(A(4,0),B(0,4),C(4,4))",
            "C(0,0) <- mma(A(0,4),B(4,0),C(0,0))",
            "C(4,0) <- mma(A(4,4),B(4,0),C(4,0))",
            "C(0,4) <- mma(A(0,4),B(4,4),C(0,4))",
            "C(4,4) <- mma(A(4,4),B(4,4),C(4,4))",
        ]
        #print("\n".join(mmgen.generate()))
        self.assertEqual(expected_sequence, list(map(str,mmgen.generate())))
