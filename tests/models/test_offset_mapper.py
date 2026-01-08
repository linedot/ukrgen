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

from ukrgen.models.lsc.offset import lsc_offset,stridexvlen
from ukrgen.models.offset_mapper import strided_mapper

class test_strided_mapper(unittest.TestCase):
    def test_cc_41(self):
        mapper = strided_mapper((4,1), (None,None))

        vtile = tile(vla_vector,scalar_dp)

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,0)),
                lsc_offset.zero_offset())
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,0)),
                lsc_offset({}, [], [1], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,0)),
                lsc_offset({}, [], [2], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,0)),
                lsc_offset({}, [], [3], 0))

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,1)),
                lsc_offset({}, [], [4], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,2)),
                lsc_offset({}, [], [8], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,3)),
                lsc_offset({}, [], [12], 0))
        
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,1)),
                lsc_offset({}, [], [5], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,2)),
                lsc_offset({}, [], [10], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,3)),
                lsc_offset({}, [], [15], 0))

    def test_rr_41(self):
        mapper = strided_mapper((1,4), (None,None), vecdim=1)

        vtile = tile(scalar_dp,vla_vector)

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,0)),
                lsc_offset.zero_offset())
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,0)),
                lsc_offset({}, [], [4], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,0)),
                lsc_offset({}, [], [8], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,0)),
                lsc_offset({}, [], [12], 0))

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,1)),
                lsc_offset({}, [], [1], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,2)),
                lsc_offset({}, [], [2], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,3)),
                lsc_offset({}, [], [3], 0))
        
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,1)),
                lsc_offset({}, [], [5], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,2)),
                lsc_offset({}, [], [10], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,3)),
                lsc_offset({}, [], [15], 0))


    def test_gc_41(self):
        mapper = strided_mapper((4,1), (0,None))

        vtile = tile(vla_vector,scalar_dp)

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,0)),
                lsc_offset.zero_offset())
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,0)),
                lsc_offset({stridexvlen({0}, {0}) : 1}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,0)),
                lsc_offset({stridexvlen({0}, {0}) : 2}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,0)),
                lsc_offset({stridexvlen({0}, {0}) : 3}, [], [], 0))

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,1)),
                lsc_offset({stridexvlen({0}, {0}) : 4}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,2)),
                lsc_offset({stridexvlen({0}, {0}) : 8}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,3)),
                lsc_offset({stridexvlen({0}, {0}) : 12}, [], [], 0))
        
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,1)),
                lsc_offset({stridexvlen({0}, {0}) : 5}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,2)),
                lsc_offset({stridexvlen({0}, {0}) : 10}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,3)),
                lsc_offset({stridexvlen({0}, {0}) : 15}, [], [], 0))


    def test_gg_41(self):
        mapper = strided_mapper((4,1), (0,1))

        vtile = tile(vla_vector,scalar_dp)

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,0)),
                lsc_offset.zero_offset())
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,0)),
                lsc_offset({stridexvlen({0}, {0}) : 1}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,0)),
                lsc_offset({stridexvlen({0}, {0}) : 2}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,0)),
                lsc_offset({stridexvlen({0}, {0}) : 3}, [], [], 0))

        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,1)),
                lsc_offset({}, [0,1], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,2)),
                lsc_offset({}, [0,2], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (0,3)),
                lsc_offset({}, [0,3], [], 0))
        
        self.assertEqual(
                mapper.map_tile_idx(vtile, (1,1)),
                lsc_offset({}, [0,1], [], 0) +
                lsc_offset({stridexvlen({0}, {0}) : 1}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (2,2)),
                lsc_offset({}, [0,2], [], 0) +
                lsc_offset({stridexvlen({0}, {0}) : 2}, [], [], 0))
        self.assertEqual(
                mapper.map_tile_idx(vtile, (3,3)),
                lsc_offset({}, [0,3], [], 0) +
                lsc_offset({stridexvlen({0}, {0}) : 3}, [], [], 0))

