# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.models.load_store_operations import (
        lsc_load,
        lsc_addr_add,
        lsc_transformation
        )
from ukrgen.models.lsc.reg import lsc_reg_type as lrt
from ukrgen.schedulers.dependency import dependency_type as dept
from ukrgen.schedulers.distance import distance_specification as dspec

class test_distance_specification(unittest.TestCase):
    def test_from_string_raw_vreg_6(self):
        dsstr = "raw::vreg:::6"

        ds = dspec.from_string(specstr=dsstr)

        self.assertEqual(ds.dep_type, dept.RAW)
        self.assertIsNone(ds.reg_type)
        self.assertEqual(ds.asm_reg_tag, "vreg")
        self.assertIsNone(ds.op1_type)
        self.assertIsNone(ds.op2_type)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 6)

    def test_from_string_raw_data_5(self):
        dsstr = "raw:data::::5"

        ds = dspec.from_string(specstr=dsstr)

        self.assertEqual(ds.dep_type, dept.RAW)
        self.assertEqual(ds.reg_type,lrt.data)
        self.assertIsNone(ds.asm_reg_tag)
        self.assertIsNone(ds.op1_type)
        self.assertIsNone(ds.op2_type)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 5)

    def test_from_string_war_addr_ld_add_3(self):
        dsstr = "war:address::ld:aadd:3"

        ds = dspec.from_string(specstr=dsstr)

        self.assertEqual(ds.dep_type, dept.WAR)
        self.assertEqual(ds.reg_type,lrt.address)
        self.assertIsNone(ds.asm_reg_tag)
        self.assertEqual(ds.op1_type, lsc_load)
        self.assertEqual(ds.op2_type, lsc_addr_add)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 3)

    def test_from_string_treg_8(self):
        dsstr = "::treg:::8"

        ds = dspec.from_string(specstr=dsstr)

        self.assertIsNone(ds.dep_type)
        self.assertIsNone(ds.reg_type)
        self.assertEqual(ds.asm_reg_tag, "treg")
        self.assertIsNone(ds.op1_type)
        self.assertIsNone(ds.op2_type)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 8)

    def test_from_string_all_2(self):
        dsstr = ":::::2"

        ds = dspec.from_string(specstr=dsstr)

        self.assertIsNone(ds.dep_type)
        self.assertIsNone(ds.reg_type)
        self.assertIsNone(ds.asm_reg_tag)
        self.assertIsNone(ds.op1_type)
        self.assertIsNone(ds.op2_type)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 2)

    def test_from_string_war_trf_data_5(self):
        dsstr = "war:data::trf::5"

        ds = dspec.from_string(specstr=dsstr)

        self.assertEqual(ds.dep_type,dept.WAR)
        self.assertEqual(ds.reg_type,lrt.data)
        self.assertIsNone(ds.asm_reg_tag)
        self.assertEqual(ds.op1_type, lsc_transformation)
        self.assertIsNone(ds.op2_type)
        self.assertIsNone(ds.tr1_name)
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 5)

    def test_from_string_war_fma_vreg_6(self):
        dsstr = "war::vreg:fma::6"

        ds = dspec.from_string(specstr=dsstr)

        self.assertEqual(ds.dep_type,dept.WAR)
        self.assertIsNone(ds.reg_type)
        self.assertEqual(ds.asm_reg_tag,"vreg")
        self.assertEqual(ds.op1_type, lsc_transformation)
        self.assertIsNone(ds.op2_type)
        self.assertEqual(ds.tr1_name, "fma")
        self.assertIsNone(ds.tr2_name)
        self.assertEqual(ds.distance, 6)
