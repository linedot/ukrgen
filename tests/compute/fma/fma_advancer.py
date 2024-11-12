from algobuild.compute.fma import fma_advancer, fma_vector_mxn_afirst_advancer, fma_vector_mxn_bfirst_advancer

import unittest

from typing import Type

class fma_advancer_test(unittest.TestCase):

    def _test_cleanrotate(self, acount : int, bcount : int, ccount : int,
                          m : int, n : int, steps : int,
                          advancer_type : Type[fma_advancer]):
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

        for i in range(steps):
            a = fma_advancer.a_vector
            b = fma_advancer.b_vector
            c = fma_advancer.c_vector
            fma_advancer.advance()

        self.assertEqual(fma_advancer.a_vector, a_vectors[0])
        self.assertEqual(fma_advancer.b_vector, b_vectors[0])
        self.assertEqual(fma_advancer.c_vector, c_vectors[0])

    def test_2x3_3_4_72_afirst(self):
        self._test_cleanrotate(acount=3, bcount=4, ccount=6,
                               m=2, n=3, steps=6*12,
                               advancer_type=fma_vector_mxn_afirst_advancer)

    def test_2x3_3_5_90_afirst(self):
        self._test_cleanrotate(acount=3, bcount=5, ccount=6,
                               m=2, n=3, steps=6*15,
                               advancer_type=fma_vector_mxn_afirst_advancer)

    def test_4x6_8_1_8_afirst(self):
        self._test_cleanrotate(acount=8, bcount=1, ccount=24,
                               m=4, n=6, steps=24*2,
                               advancer_type=fma_vector_mxn_afirst_advancer)

    def test_4x6_6_1_8_afirst(self):
        self._test_cleanrotate(acount=6, bcount=1, ccount=24,
                               m=4, n=6, steps=24*3,
                               advancer_type=fma_vector_mxn_afirst_advancer)

    def test_2x3_3_4_72_bfirst(self):
        self._test_cleanrotate(acount=3, bcount=4, ccount=6,
                               m=2, n=3, steps=6*12,
                               advancer_type=fma_vector_mxn_bfirst_advancer)

    def test_2x3_3_5_90_bfirst(self):
        self._test_cleanrotate(acount=3, bcount=5, ccount=6,
                               m=2, n=3, steps=6*15,
                               advancer_type=fma_vector_mxn_bfirst_advancer)

    def test_4x6_8_1_8_bfirst(self):
        self._test_cleanrotate(acount=8, bcount=1, ccount=24,
                               m=4, n=6, steps=24*2,
                               advancer_type=fma_vector_mxn_bfirst_advancer)

    def test_4x6_6_1_8_bfirst(self):
        self._test_cleanrotate(acount=6, bcount=1, ccount=24,
                               m=4, n=6, steps=24*3,
                               advancer_type=fma_vector_mxn_bfirst_advancer)



