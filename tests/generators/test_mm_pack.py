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
from ukrgen.generators.mm import mm,order2D
from ukrgen.models import load_store_cpu
from ukrgen.models.offset_mapper import flat_mapper
from ukrgen.models.load_store_operations import lsc_offset
from ukrgen.models.addr_resolver import addr_resolver
from ukrgen.schedulers import simple_dependency_scheduler


class test_mm_pack(unittest.TestCase):

    def test_2vx4_kappag(self):
        m = 2
        n = 4
        unroll_factor = 1
        k = unroll_factor

        pack_tile = simple_ukr_tile(a_size=m,
                                    b_size=n,
                                    subdims=(vla_vector,
                                             scalar_dp))
        scale_tile = simple_ukr_tile(a_size=n, b_size=n,
                                     subdims=(scalar_dp,scalar_dp),
                                     bands=(0,0))
        genpack = mm(pack_tile, scale_tile, pack_tile,
                     lo=order2D("kKmMnN"),
                     opstr="fmul", tile_strs=["X","kappa","Y"])

        pack_ops = genpack.generate()
        pack_ops_next = genpack.generate(add_dims=[0,0,0,0,n,n])

        expected_pack_ops=[
            "Y(0*VLEN,0) <- fmul(X(0*VLEN,0),kappa(0,0),Y(0*VLEN,0))",
            "Y(1*VLEN,0) <- fmul(X(1*VLEN,0),kappa(0,0),Y(1*VLEN,0))",
            "Y(0*VLEN,1) <- fmul(X(0*VLEN,1),kappa(1,1),Y(0*VLEN,1))",
            "Y(1*VLEN,1) <- fmul(X(1*VLEN,1),kappa(1,1),Y(1*VLEN,1))",
            "Y(0*VLEN,2) <- fmul(X(0*VLEN,2),kappa(2,2),Y(0*VLEN,2))",
            "Y(1*VLEN,2) <- fmul(X(1*VLEN,2),kappa(2,2),Y(1*VLEN,2))",
            "Y(0*VLEN,3) <- fmul(X(0*VLEN,3),kappa(3,3),Y(0*VLEN,3))",
            "Y(1*VLEN,3) <- fmul(X(1*VLEN,3),kappa(3,3),Y(1*VLEN,3))",
        ]
        expected_pack_ops_next=[
            "Y(0*VLEN,4) <- fmul(X(0*VLEN,4),kappa(4,4),Y(0*VLEN,4))",
            "Y(1*VLEN,4) <- fmul(X(1*VLEN,4),kappa(4,4),Y(1*VLEN,4))",
            "Y(0*VLEN,5) <- fmul(X(0*VLEN,5),kappa(5,5),Y(0*VLEN,5))",
            "Y(1*VLEN,5) <- fmul(X(1*VLEN,5),kappa(5,5),Y(1*VLEN,5))",
            "Y(0*VLEN,6) <- fmul(X(0*VLEN,6),kappa(6,6),Y(0*VLEN,6))",
            "Y(1*VLEN,6) <- fmul(X(1*VLEN,6),kappa(6,6),Y(1*VLEN,6))",
            "Y(0*VLEN,7) <- fmul(X(0*VLEN,7),kappa(7,7),Y(0*VLEN,7))",
            "Y(1*VLEN,7) <- fmul(X(1*VLEN,7),kappa(7,7),Y(1*VLEN,7))",
        ]
        self.assertEqual(expected_pack_ops,
                         list(map(str,pack_ops)))
        self.assertEqual(expected_pack_ops_next,
                         list(map(str,pack_ops_next)))

    def test_2vx4_kappa1(self):
        m = 2
        n = 4
        unroll_factor = 1
        k = unroll_factor

        pack_tile = simple_ukr_tile(a_size=m,
                                    b_size=n,
                                    subdims=(vla_vector,
                                             scalar_dp))
        scale_tile = simple_ukr_tile(a_size=n, b_size=n,
                                     subdims=(scalar_dp,scalar_dp),
                                     bands=(0,0))
        genpack = mm(pack_tile, scale_tile, pack_tile,
                     lo=order2D("kKmMnN"),
                     opstr="get1", tile_strs=["X","kappa","Y"])

        pack_ops = genpack.generate()
        pack_ops_next = genpack.generate(add_dims=[0,0,0,0,n,n])

        expected_pack_ops=[
            "Y(0*VLEN,0) <- get1(X(0*VLEN,0),kappa(0,0),Y(0*VLEN,0))",
            "Y(1*VLEN,0) <- get1(X(1*VLEN,0),kappa(0,0),Y(1*VLEN,0))",
            "Y(0*VLEN,1) <- get1(X(0*VLEN,1),kappa(1,1),Y(0*VLEN,1))",
            "Y(1*VLEN,1) <- get1(X(1*VLEN,1),kappa(1,1),Y(1*VLEN,1))",
            "Y(0*VLEN,2) <- get1(X(0*VLEN,2),kappa(2,2),Y(0*VLEN,2))",
            "Y(1*VLEN,2) <- get1(X(1*VLEN,2),kappa(2,2),Y(1*VLEN,2))",
            "Y(0*VLEN,3) <- get1(X(0*VLEN,3),kappa(3,3),Y(0*VLEN,3))",
            "Y(1*VLEN,3) <- get1(X(1*VLEN,3),kappa(3,3),Y(1*VLEN,3))",
        ]
        expected_pack_ops_next=[
            "Y(0*VLEN,4) <- get1(X(0*VLEN,4),kappa(4,4),Y(0*VLEN,4))",
            "Y(1*VLEN,4) <- get1(X(1*VLEN,4),kappa(4,4),Y(1*VLEN,4))",
            "Y(0*VLEN,5) <- get1(X(0*VLEN,5),kappa(5,5),Y(0*VLEN,5))",
            "Y(1*VLEN,5) <- get1(X(1*VLEN,5),kappa(5,5),Y(1*VLEN,5))",
            "Y(0*VLEN,6) <- get1(X(0*VLEN,6),kappa(6,6),Y(0*VLEN,6))",
            "Y(1*VLEN,6) <- get1(X(1*VLEN,6),kappa(6,6),Y(1*VLEN,6))",
            "Y(0*VLEN,7) <- get1(X(0*VLEN,7),kappa(7,7),Y(0*VLEN,7))",
            "Y(1*VLEN,7) <- get1(X(1*VLEN,7),kappa(7,7),Y(1*VLEN,7))",
        ]
        self.assertEqual(expected_pack_ops,
                         list(map(str,pack_ops)))
        self.assertEqual(expected_pack_ops_next,
                         list(map(str,pack_ops_next)))
