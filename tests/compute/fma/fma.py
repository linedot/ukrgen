from algobuild.compute.fma import (
        fma,
        fma_info,
        vector_vector_fma,
        fma_advancer, 
        fma_vector_mxn_afirst_advancer, 
        fma_vector_mxn_bfirst_advancer
        )
from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.avx_fma import fma128
from asmgen.asmblocks.avx_fma import fma256
from asmgen.asmblocks.avx_fma import avx512

from parameterized import parameterized

import unittest

from typing import Type

all_generators = [
    ("rvv", rvv()),
    ("rvv071", rvv071()),
    ("neon", neon()),
    ("sve", sve()),
    ("fma128", fma128()),
    ("fma256", fma256()),
    ("fma512", avx512()),
]
def withoutgen(generators : list[tuple[str,asmgen]], without : list[str]):
    return [gen for gen in generators if gen[0] not in without]

class fma_vector_vector_test(unittest.TestCase):

    def _generate_block(self, acount : int, bcount : int, ccount : int,
                          m : int, n : int, steps : int,
                          advancer_type : Type[fma_advancer],
                          fma : fma):
        a_vectors=[i+0 for i in range(acount)]
        b_vectors=[i+acount for i in range(bcount)]
        c_vectors=[i+acount+bcount for i in range(ccount)]
        if advancer_type == fma_vector_mxn_bfirst_advancer:
            c_vectors = [(i*m)%ccount+(i*m)//ccount for i in range(ccount)]
            c_vectors = [i+acount+bcount for i in c_vectors]
        fma_advancer = advancer_type(
            a_vectors=a_vectors,
            b_vectors=b_vectors,
            c_vectors=c_vectors,
            m=m, n=n)

        asmblock = ""
        for i in range(steps):
            a = fma_advancer.a_vector
            b = fma_advancer.b_vector
            c = fma_advancer.c_vector
            fma.fma_info.a_data_register = a
            fma.fma_info.b_data_register = b
            fma.fma_info.c_data_register = c

            asmblock += fma.process()

            fma_advancer.advance()

        return asmblock


    # Note: If one generator fails but another doesn't, something
    #       really weird is happening, because the 'asmgen.fma()'
    #       method is used by the vector_vector_fma class
    @parameterized.expand(all_generators)
    def test_2x3_3_4_72_afirst(self, name : str, asmgen : asmgen):
        fi = fma_info()
        dt = fi.datatype
        fma = vector_vector_fma(asmgen=asmgen,fma_info=fi)
        asmblock = self._generate_block(acount=3, bcount=4, ccount=6,
                               m=2, n=3, steps=6*12,
                               advancer_type=fma_vector_mxn_afirst_advancer,
                               fma=fma)

        v = asmgen.vreg
        expected_commands=[
            asmgen.fma(v(0),v(3),v(7),dt),
            asmgen.fma(v(1),v(3),v(8),dt),
            asmgen.fma(v(0),v(4),v(9),dt),
            asmgen.fma(v(1),v(4),v(10),dt),
            asmgen.fma(v(0),v(5),v(11),dt),
            asmgen.fma(v(1),v(5),v(12),dt),
            asmgen.fma(v(2),v(6),v(7),dt),
            asmgen.fma(v(0),v(6),v(8),dt),
            asmgen.fma(v(2),v(3),v(9),dt),
            asmgen.fma(v(0),v(3),v(10),dt),
            asmgen.fma(v(2),v(4),v(11),dt),
            asmgen.fma(v(0),v(4),v(12),dt),
            asmgen.fma(v(1),v(5),v(7),dt),
            asmgen.fma(v(2),v(5),v(8),dt),
            asmgen.fma(v(1),v(6),v(9),dt),
            asmgen.fma(v(2),v(6),v(10),dt),
            asmgen.fma(v(1),v(3),v(11),dt),
            asmgen.fma(v(2),v(3),v(12),dt),
            asmgen.fma(v(0),v(4),v(7),dt),
            asmgen.fma(v(1),v(4),v(8),dt),
            asmgen.fma(v(0),v(5),v(9),dt),
            asmgen.fma(v(1),v(5),v(10),dt),
            asmgen.fma(v(0),v(6),v(11),dt),
            asmgen.fma(v(1),v(6),v(12),dt),
            asmgen.fma(v(2),v(3),v(7),dt),
            asmgen.fma(v(0),v(3),v(8),dt),
            asmgen.fma(v(2),v(4),v(9),dt),
            asmgen.fma(v(0),v(4),v(10),dt),
            asmgen.fma(v(2),v(5),v(11),dt),
            asmgen.fma(v(0),v(5),v(12),dt),
            asmgen.fma(v(1),v(6),v(7),dt),
            asmgen.fma(v(2),v(6),v(8),dt),
            asmgen.fma(v(1),v(3),v(9),dt),
            asmgen.fma(v(2),v(3),v(10),dt),
            asmgen.fma(v(1),v(4),v(11),dt),
            asmgen.fma(v(2),v(4),v(12),dt),
            asmgen.fma(v(0),v(5),v(7),dt),
            asmgen.fma(v(1),v(5),v(8),dt),
            asmgen.fma(v(0),v(6),v(9),dt),
            asmgen.fma(v(1),v(6),v(10),dt),
            asmgen.fma(v(0),v(3),v(11),dt),
            asmgen.fma(v(1),v(3),v(12),dt),
            asmgen.fma(v(2),v(4),v(7),dt),
            asmgen.fma(v(0),v(4),v(8),dt),
            asmgen.fma(v(2),v(5),v(9),dt),
            asmgen.fma(v(0),v(5),v(10),dt),
            asmgen.fma(v(2),v(6),v(11),dt),
            asmgen.fma(v(0),v(6),v(12),dt),
            asmgen.fma(v(1),v(3),v(7),dt),
            asmgen.fma(v(2),v(3),v(8),dt),
            asmgen.fma(v(1),v(4),v(9),dt),
            asmgen.fma(v(2),v(4),v(10),dt),
            asmgen.fma(v(1),v(5),v(11),dt),
            asmgen.fma(v(2),v(5),v(12),dt),
            asmgen.fma(v(0),v(6),v(7),dt),
            asmgen.fma(v(1),v(6),v(8),dt),
            asmgen.fma(v(0),v(3),v(9),dt),
            asmgen.fma(v(1),v(3),v(10),dt),
            asmgen.fma(v(0),v(4),v(11),dt),
            asmgen.fma(v(1),v(4),v(12),dt),
            asmgen.fma(v(2),v(5),v(7),dt),
            asmgen.fma(v(0),v(5),v(8),dt),
            asmgen.fma(v(2),v(6),v(9),dt),
            asmgen.fma(v(0),v(6),v(10),dt),
            asmgen.fma(v(2),v(3),v(11),dt),
            asmgen.fma(v(0),v(3),v(12),dt),
            asmgen.fma(v(1),v(4),v(7),dt),
            asmgen.fma(v(2),v(4),v(8),dt),
            asmgen.fma(v(1),v(5),v(9),dt),
            asmgen.fma(v(2),v(5),v(10),dt),
            asmgen.fma(v(1),v(6),v(11),dt),
            asmgen.fma(v(2),v(6),v(12),dt),
        ]

        expected_block = "".join(expected_commands)
        self.assertEqual(asmblock,expected_block)

    @parameterized.expand(withoutgen(all_generators,["fma128","fma256"]))
    def test_4x6_6_2_72_afirst(self, name : str, asmgen : asmgen):
        fi = fma_info()
        dt = fi.datatype
        fma = vector_vector_fma(asmgen=asmgen,fma_info=fi)
        asmblock = self._generate_block(acount=6, bcount=2, ccount=24,
                               m=4, n=6, steps=72,
                               advancer_type=fma_vector_mxn_afirst_advancer,
                               fma=fma)

        v = asmgen.vreg
        expected_commands=[
            asmgen.fma(v(0),v(6),v(8),dt),
            asmgen.fma(v(1),v(6),v(9),dt),
            asmgen.fma(v(2),v(6),v(10),dt),
            asmgen.fma(v(3),v(6),v(11),dt),
            asmgen.fma(v(0),v(7),v(12),dt),
            asmgen.fma(v(1),v(7),v(13),dt),
            asmgen.fma(v(2),v(7),v(14),dt),
            asmgen.fma(v(3),v(7),v(15),dt),
            asmgen.fma(v(0),v(6),v(16),dt),
            asmgen.fma(v(1),v(6),v(17),dt),
            asmgen.fma(v(2),v(6),v(18),dt),
            asmgen.fma(v(3),v(6),v(19),dt),
            asmgen.fma(v(0),v(7),v(20),dt),
            asmgen.fma(v(1),v(7),v(21),dt),
            asmgen.fma(v(2),v(7),v(22),dt),
            asmgen.fma(v(3),v(7),v(23),dt),
            asmgen.fma(v(0),v(6),v(24),dt),
            asmgen.fma(v(1),v(6),v(25),dt),
            asmgen.fma(v(2),v(6),v(26),dt),
            asmgen.fma(v(3),v(6),v(27),dt),
            asmgen.fma(v(0),v(7),v(28),dt),
            asmgen.fma(v(1),v(7),v(29),dt),
            asmgen.fma(v(2),v(7),v(30),dt),
            asmgen.fma(v(3),v(7),v(31),dt),

            asmgen.fma(v(4),v(6),v(8),dt),
            asmgen.fma(v(5),v(6),v(9),dt),
            asmgen.fma(v(0),v(6),v(10),dt),
            asmgen.fma(v(1),v(6),v(11),dt),
            asmgen.fma(v(4),v(7),v(12),dt),
            asmgen.fma(v(5),v(7),v(13),dt),
            asmgen.fma(v(0),v(7),v(14),dt),
            asmgen.fma(v(1),v(7),v(15),dt),
            asmgen.fma(v(4),v(6),v(16),dt),
            asmgen.fma(v(5),v(6),v(17),dt),
            asmgen.fma(v(0),v(6),v(18),dt),
            asmgen.fma(v(1),v(6),v(19),dt),
            asmgen.fma(v(4),v(7),v(20),dt),
            asmgen.fma(v(5),v(7),v(21),dt),
            asmgen.fma(v(0),v(7),v(22),dt),
            asmgen.fma(v(1),v(7),v(23),dt),
            asmgen.fma(v(4),v(6),v(24),dt),
            asmgen.fma(v(5),v(6),v(25),dt),
            asmgen.fma(v(0),v(6),v(26),dt),
            asmgen.fma(v(1),v(6),v(27),dt),
            asmgen.fma(v(4),v(7),v(28),dt),
            asmgen.fma(v(5),v(7),v(29),dt),
            asmgen.fma(v(0),v(7),v(30),dt),
            asmgen.fma(v(1),v(7),v(31),dt),

            asmgen.fma(v(2),v(6),v(8),dt),
            asmgen.fma(v(3),v(6),v(9),dt),
            asmgen.fma(v(4),v(6),v(10),dt),
            asmgen.fma(v(5),v(6),v(11),dt),
            asmgen.fma(v(2),v(7),v(12),dt),
            asmgen.fma(v(3),v(7),v(13),dt),
            asmgen.fma(v(4),v(7),v(14),dt),
            asmgen.fma(v(5),v(7),v(15),dt),
            asmgen.fma(v(2),v(6),v(16),dt),
            asmgen.fma(v(3),v(6),v(17),dt),
            asmgen.fma(v(4),v(6),v(18),dt),
            asmgen.fma(v(5),v(6),v(19),dt),
            asmgen.fma(v(2),v(7),v(20),dt),
            asmgen.fma(v(3),v(7),v(21),dt),
            asmgen.fma(v(4),v(7),v(22),dt),
            asmgen.fma(v(5),v(7),v(23),dt),
            asmgen.fma(v(2),v(6),v(24),dt),
            asmgen.fma(v(3),v(6),v(25),dt),
            asmgen.fma(v(4),v(6),v(26),dt),
            asmgen.fma(v(5),v(6),v(27),dt),
            asmgen.fma(v(2),v(7),v(28),dt),
            asmgen.fma(v(3),v(7),v(29),dt),
            asmgen.fma(v(4),v(7),v(30),dt),
            asmgen.fma(v(5),v(7),v(31),dt),
        ]

        expected_block = "".join(expected_commands)
        self.maxDiff = None
        self.assertEqual(asmblock,expected_block)
