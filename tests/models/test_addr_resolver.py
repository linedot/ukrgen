# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest
import string

from ukrgen.models.load_store_operations import lsc_offset
from ukrgen.models.addr_resolver import addr_resolver,addr_add

class test_addr_resolver(unittest.TestCase):
    def test_zerorange(self):
        zo = lsc_offset.zero_offset()

        ar = addr_resolver(indices=[[0,1]],
                           starting_offsets=[[zo,zo]],
                           offset_ranges=[[(zo,zo),
                                           (zo,zo)]],
                           steps=[[zo,zo]])

        addr_adds,idx,off = ar.resolve_addr(0, zo)

        self.assertFalse(addr_adds)
        self.assertEqual(0, idx)
        self.assertEqual(zo, off)

    def test_s2v_r1v(self):

        zo = lsc_offset.zero_offset()
        o2v = lsc_offset({}, [], [2], 0)
        o1v = lsc_offset({}, [], [1], 0)

        ar = addr_resolver(indices=[[0]],
                           starting_offsets=[[zo]],
                           offset_ranges=[[(zo,o1v)]],
                           steps=[[o2v]])


        add_o2v = addr_add(0, 0, o2v)
        expected_tuples=[
            ([],0,zo),
            ([],0,o1v),
            ([add_o2v],0,zo),
            ([],0,o1v),
            ([add_o2v],0,zo),
            ([],0,o1v),
            ([add_o2v],0,zo),
            ([],0,o1v),
        ]

        for i in range(0,8):
            toff = lsc_offset({}, [], [i], 0)

            addr_adds,idx,off = ar.resolve_addr(0, toff)
            e_addr_adds,e_idx,e_off = expected_tuples[i]

            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)


    def test_a2_s4v_r1v_so2v(self):

        zo = lsc_offset.zero_offset()
        o4v = lsc_offset({},[], [4], 0)
        o2v = lsc_offset({},[], [2], 0)
        o1v = lsc_offset({},[], [1], 0)

        ar = addr_resolver(indices=[[0,1]],
                           starting_offsets=[[zo,o2v]],
                           offset_ranges=[[(zo,o1v)]*2],
                           steps=[[o4v]*2])


        add0_o4v = addr_add(0, 0, o4v)
        add1_o4v = addr_add(0, 1, o4v)
        expected_tuples=[
            ([],0,zo),
            ([],0,o1v),
            ([],1,zo),
            ([],1,o1v),
            ([add0_o4v],0,zo),
            ([],0,o1v),
            ([add1_o4v],1,zo),
            ([],1,o1v),
        ]

        for i in range(0,8):
            toff = lsc_offset({},[], [i], 0)

            addr_adds,idx,off = ar.resolve_addr(0, toff)
            e_addr_adds,e_idx,e_off = expected_tuples[i]

            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)

    def test_a4_s8v_r1v_so2v(self):

        zo = lsc_offset.zero_offset()
        o8v = lsc_offset({},[], [8], 0)
        o6v = lsc_offset({},[], [6], 0)
        o4v = lsc_offset({},[], [4], 0)
        o2v = lsc_offset({},[], [2], 0)
        o1v = lsc_offset({},[], [1], 0)

        ar = addr_resolver(indices=[[0,1,2,3]],
                           starting_offsets=[[zo,o2v,o4v,o6v]],
                           offset_ranges=[[(zo,o1v)]*4],
                           steps=[[o8v]*4])


        add0_o4v = addr_add(0, 0, o4v)
        add1_o4v = addr_add(0, 1, o4v)
        expected_tuples=[
            ([],0,zo),
            ([],0,o1v),
            ([],1,zo),
            ([],1,o1v),
            ([],2,zo),
            ([],2,o1v),
            ([],3,zo),
            ([],3,o1v),
        ]

        for i in range(0,8):
            toff = lsc_offset({},[], [i], 0)

            addr_adds,idx,off = ar.resolve_addr(0, toff)
            e_addr_adds,e_idx,e_off = expected_tuples[i]

            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)

    def test_a4_s8v_r1v_so2v_16iter(self):

        zo = lsc_offset.zero_offset()
        o8v = lsc_offset({},[], [8], 0)
        o6v = lsc_offset({},[], [6], 0)
        o4v = lsc_offset({},[], [4], 0)
        o2v = lsc_offset({},[], [2], 0)
        o1v = lsc_offset({},[], [1], 0)

        ar = addr_resolver(indices=[[0,1,2,3]],
                           starting_offsets=[[zo,o2v,o4v,o6v]],
                           offset_ranges=[[(zo,o1v)]*4],
                           steps=[[o8v]*4])


        add0_o8v = addr_add(0, 0, o8v)
        add1_o8v = addr_add(0, 1, o8v)
        add2_o8v = addr_add(0, 2, o8v)
        add3_o8v = addr_add(0, 3, o8v)
        expected_tuples=[
            ([],0,zo),
            ([],0,o1v),
            ([],1,zo),
            ([],1,o1v),
            ([],2,zo),
            ([],2,o1v),
            ([],3,zo),
            ([],3,o1v),
            ([add0_o8v],0,zo),
            ([],0,o1v),
            ([add1_o8v],1,zo),
            ([],1,o1v),
            ([add2_o8v],2,zo),
            ([],2,o1v),
            ([add3_o8v],3,zo),
            ([],3,o1v),
        ]

        for i in range(0,8):
            toff = lsc_offset({},[], [i], 0)

            addr_adds,idx,off = ar.resolve_addr(0, toff)
            e_addr_adds,e_idx,e_off = expected_tuples[i]

            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)

    def test_abc_a211_s4v2i2v_r3v1i1v_so2v1i1v(self):

        zo = lsc_offset.zero_offset()
        o8v = lsc_offset({},[], [8], 0)
        o6v = lsc_offset({},[], [6], 0)
        o4v = lsc_offset({},[], [4], 0)
        o2v = lsc_offset({},[], [2], 0)
        o1v = lsc_offset({},[], [1], 0)

        o2i = lsc_offset({},[],[],2)
        o1i = lsc_offset({},[],[],1)

        ar = addr_resolver(indices=[[0,1],[0],[0]],
                           starting_offsets=[[zo,o2v],[zo],[zo]],
                           offset_ranges=[[(zo,o1v)]*2,[(zo,o1i)],[(zo,o1i)]],
                           steps=[[o4v]*2,[o2i],[o2i]])


        add0_o4v = addr_add(0, 0, o4v)
        add1_o4v = addr_add(0, 1, o4v)
        expected_a_tuples=[
            ([],0,zo),
            ([],0,o1v),
            ([],1,zo),
            ([],1,o1v),
            ([add0_o4v],0,zo),
            ([],0,o1v),
            ([add1_o4v],1,zo),
            ([],1,o1v)
        ]

        add0_o2i = addr_add(1, 0, o2i)
        expected_b_tuples=[
            ([],0,zo),
            ([],0,o1i),
            ([add0_o2i],0,zo),
            ([],0,o1i),
            ([add0_o2i],0,zo),
            ([],0,o1i),
            ([add0_o2i],0,zo),
            ([],0,o1i),
        ]

        add0_o2i = addr_add(2, 0, o2i)
        expected_c_tuples=[
            ([],0,zo),
            ([],0,zo),
            ([],0,o1i),
            ([],0,o1i),
            ([add0_o2i],0,zo),
            ([],0,zo),
            ([],0,o1i),
            ([],0,o1i),
        ]

        for i in range(0,8):
            toff = lsc_offset({},[], [i], 0)
            addr_adds,idx,off = ar.resolve_addr(0, toff)
            rtype_char = string.ascii_lowercase[0]
            e_addr_adds,e_idx,e_off = expected_a_tuples[i]
            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)

            toff = lsc_offset({},[], [], i)
            addr_adds,idx,off = ar.resolve_addr(1, toff)
            rtype_char = string.ascii_lowercase[1]
            e_addr_adds,e_idx,e_off = expected_b_tuples[i]
            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)

            toff = lsc_offset({},[], [], i//2)
            addr_adds,idx,off = ar.resolve_addr(2, toff)
            rtype_char = string.ascii_lowercase[2]
            e_addr_adds,e_idx,e_off = expected_c_tuples[i]
            self.assertEqual(addr_adds, e_addr_adds)
            self.assertEqual(idx, e_idx)
            self.assertEqual(off, e_off)
