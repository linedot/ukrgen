# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .stage import stage
from .unvec import unvec_stage
from .ukr import ukr_composition_map
from ..ukr_context import ukr_context
from ..stage_param import stage_param
from ...specializers.asm import op_support
from ...components.tile import copy_with_vecdir

class geometry_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)
        

        ukr = self.context.params["ukr"].value

        composition = ukr_composition_map[ukr]

        dimensions = set()
        # get dimensions by investigating all STO descriptions
        for sto in composition.get_sto_descriptions():
            for dims in sto.dimensions.values():
                dimensions.update(dims)

        dimensions = sorted(list(dimensions))


    def progress(self) -> list[stage]:


        # tile assignment logic from the specializer:
        # if arith_op.widening_method == wm.DOT_NEIGHBOURS and\
        #   op == 'fopa' and\
        #   ways > 1:
        #     # This makes things more complicated for offset computation
        #     #n_dp = dimension_properties(dt=dimension_type.fixed, size=ways,
        #     #                            sdt=dimension_type.fixed, sd_size=ways)
        #     #a_tile,b_tile,c_tile = tile(vec_dp, n_dp), tile(n_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
        #     a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(scalar_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
        # elif op == 'fopa':
        #     a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(scalar_dp, vec_dp), tile(wide_vec_dp, wide_vec_dp)
        # elif op == 'fma' or op == 'fmul':
        #     a_tile,b_tile,c_tile = tile(vec_dp, scalar_dp), tile(vec_dp, scalar_dp), tile(wide_vec_dp, scalar_dp)
        # elif op == 'dota':
        #     a_tile,b_tile,c_tile = tile(scalar_dp, vec_dp), tile(vec_dp, scalar_dp), tile(scalar_dp, scalar_dp)
        # else:
        #     raise NotImplementedError(f"Adding tiles for {op} not implemented yet")

        # if mod.VF in modifiers:
        #     b_tile = tile(scalar_dp, scalar_dp)

        # return a_tile,b_tile,c_tile

        
        # Realistically there is a limited number of things that can happen:
        #
        # Case 1: fma.vf
        # sup: A(v,s), B(s,s), C(v,s)
        #  - A(v,s), B(s,s), C(v,s)
        #  - A(s,s), B(s,v), C(s,v)
        #     - transformations:
        #     - A' = B
        #     - B' = A^T
        #     - C' = C^T
        #
        # Case 2: fma.vv
        # sup: A(v,s), B(v,s), C(v,s)
        #  - HW: A(v,s), B(v,s), C(v,s)
        #  - data:  A(v,s), B(s,s), C(v,s)
        #    - identical to  case 1
        #
        # this makes the case for unvec to happen before this step
        # Also data will always have one scalar tile, the difference
        # in "unvec methods" is just how the data is sourced
        # 
        # Case 3: dot - no variants, only lane vs scalar
        # Case 4: opa - no variants
        # Case 5: mma
        # sup: A(v1,v2), B(v2,v3), C(v1,v3)
        #  - A(v1,v2), B(v2,v3), C(v1,v3)
        #  - A(v3,v2), B(v2,v1), C(v3,v1)
        #     - transformations:
        #     - A' = B^T
        #     - B' = A^T
        #     - C' = C^T
        #
        # There might be something else with RISC-V IME
        #
        # This is for the GEMM kernel where each TO has
        # A(m,k), B(k,n), C(m,n)
        #
        # For scale/pack we have:
        # A(m,n), B(n,n), C(m,n)
        # something like
        # A(n,m), B(m,m), C(n,m)
        # would be possible, since B is just a scalar wearing a matrix disguise,
        # but not sure how to express this and if it's useful


        # There might be some kind of general case here, but for now just
        # allow 
        #  - A' = B^T (equivalent to B in scalar case)
        #  - B' = A^T
        #  - C' = C^T
        #
        # if C.dim1 == A.dim1 and C.dim2 == B.dim2
        



        self.context.params.update(self.params)


        vecdir = self.context.params["vecdir"].value
        assert vecdir in ["M","N"], f"Invalid vecdir: {vecdir}"


        # TODO: There should be some kind of generalization
        # TODO: real/data tile are already used in some places,
        #       maybe there should be support/component tiles?
        b_map = {"M" : -1, "N" : 1}
        a_map = {"M" : 0, "N" : -1}
        c_map = {"M" : 0, "N" : 1}

        if vecdir == "N":
            self.context.sup.b_tile,self.context.sup.a_tile = \
              self.context.sup.a_tile,self.context.sup.b_tile
    
        if -1 != b_map[vecdir]:
            self.context.sup.b_tile = copy_with_vecdir(
                    t=self.context.sup.b_tile,
                    vectorized_dimension=b_map[vecdir])
        if -1 != a_map[vecdir]:
            self.context.sup.a_tile = copy_with_vecdir(
                    t=self.context.sup.a_tile,
                    vectorized_dimension=a_map[vecdir])
        if -1 != c_map[vecdir]:
            self.context.sup.c_tile = copy_with_vecdir(
                    t=self.context.sup.c_tile,
                    vectorized_dimension=c_map[vecdir])

        # Do we need to unvec?
        if self.context.params["op"].value == "fma" and \
                self.context.sup.b_tile.is_vector and \
                self.context.sup.a_tile.is_vector:

            return [unvec_stage]
        else:
            return list()
