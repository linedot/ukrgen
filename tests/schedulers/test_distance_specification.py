# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.models.load_store_operations import (
        lsc_load,
        lsc_store,
        lsc_addr_add,
        lsc_transformation
        )
from ukrgen.models.lsc.reg import lsc_reg_type as lrt, lsc_reg_index as lri
from ukrgen.models.lsc.offset import lsc_offset as lo
from ukrgen.schedulers.dependency import dependency_type as dept
from ukrgen.schedulers.distance import distance_specification as dspec

from ukrgen.components.tile import scalar_tile as stile,vla_vector_tile as vtile

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
    

    def test_war_fma_vreg_6_fma_fma(self):
        dsstr = "war::vreg:fma::6"


        mkr = lambda c,i : lri(c, indices=[i])

        
        ds = dspec.from_string(specstr=dsstr)
        op1 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",0),
                                     mkr("B",0),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])

        op2 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",1),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 6)


    def test_war_fma_vreg_6_st_fma(self):
        dsstr = "war::vreg:fma::6"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_store("C",
                        res_idx=0,
                        addr_idx=0,
                        off=zo,
                        stride=None,
                        t=vtile,
                        mods={})

        op2 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",1),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 0)

    def test_war_fma_vreg_6_fma_ld(self):
        dsstr = "war::vreg:fma::6"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",1),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])

        op2 = lsc_load("C",
                       res_idx=0,
                       addr_idx=0,
                       off=zo,
                       stride=None,
                       t=vtile,
                       mods={})


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 6)

    def test_war_fma_vreg_6_fma_ld_freg(self):
        dsstr = "war::vreg:fma::6"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",1),
                                     mkr("C",0)], 
                                 tiles=[stile,stile,stile])

        op2 = lsc_load("C",
                       res_idx=0,
                       addr_idx=0,
                       off=zo,
                       stride=None,
                       t=stile,
                       mods={})


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 0)

    def test_war_ld_addr_3_ld_add(self):
        dsstr = "war:address::ld::3"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_load("C",
                       res_idx=0,
                       addr_idx=0,
                       off=zo,
                       stride=None,
                       t=vtile,
                       mods={})
        op2 = lsc_addr_add(component="C",
                           addr_idx=0,
                           off=lo(sxv_strides={},
                                  reg_strides=[],
                                  vlen_strides=[],
                                  immoff=4), t=vtile)


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 3)


    def test_all_3_indep(self):
        dsstr = ":::::3"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",0),
                                     mkr("B",0),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])

        op2 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",1),
                                     mkr("C",1)], 
                                 tiles=[vtile,vtile,vtile])


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 0)

    def test_all_3_dep(self):
        dsstr = ":::::3"


        mkr = lambda c,i : lri(c, indices=[i])
        zo = lo.zero_offset()

        
        ds = dspec.from_string(specstr=dsstr)

        op1 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",0),
                                     mkr("B",0),
                                     mkr("C",0)], 
                                 tiles=[vtile,vtile,vtile])

        op2 = lsc_transformation(op="fma",
                                 res_indices=[
                                     mkr("A",1),
                                     mkr("B",0),
                                     mkr("C",1)], 
                                 tiles=[vtile,vtile,vtile])


        distance = ds.apply(op1=op1, op2=op2)

        self.assertEqual(distance, 3)
