import unittest


from algobuild.generators import dummy_mm_op,tile,dimension_properties,dimension_type,storage_type

scalar = dimension_properties(dt=dimension_type.fixed, size=1,
                              sdt=dimension_type.fixed, sd_size=1)

vla_vector = dimension_properties(dt=dimension_type.vla, size=1,
                                  sdt=dimension_type.fixed, sd_size=4)

x4_vector = dimension_properties(dt=dimension_type.fixed, size=4,
                                 sdt=dimension_type.fixed, sd_size=4)
class simple_ukr_tile(tile):
    def __init__(self, 
                 a_size : int,
                 b_size : int,
                 subdims : tuple[dimension_properties]):

        dima = dimension_properties(dt=dimension_type.fixed,
                                    size=a_size,
                                    sdt=dimension_type.fixed,
                                    sd_size=a_size)
        dimb = dimension_properties(dt=dimension_type.fixed,
                                    size=b_size,
                                    sdt=dimension_type.fixed,
                                    sd_size=b_size)

        super().__init__(
                dima=dima,dimb=dimb,
                subtiles = [tile(dima=subdims[0],dimb=subdims[1])],
                subtile_count_a = 1,
                subtile_count_b = 1,
                stype = storage_type.register)

class test_mm(unittest.TestCase):
    def test_2vx4_scalar(self):
        m = 2
        n = 4
        unroll_factor = 1
        k = unroll_factor
        use_fma_vf = True

        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar,scalar))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar,scalar))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="fma")
        print("\n".join(mmgen.generate()))

    def test_2vx4_vector_opa(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar,vla_vector))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,vla_vector))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="opa")
        print("\n".join(mmgen.generate()))

    def test_2vx4_vla_vf_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar,scalar))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="fma")
        print("\n".join(mmgen.generate()))

    def test_2vx4_fix_vf_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(x4_vector,scalar))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar,scalar))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(x4_vector,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="fma")
        print("\n".join(mmgen.generate()))

    def test_2vx4_idxx4_fma(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(x4_vector,scalar))
        b_tile = simple_ukr_tile(a_size=k, b_size=n//x4_vector.sd_size, subdims=(scalar,x4_vector))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(x4_vector,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="fma_idxx4")
        print("\n".join(mmgen.generate()))

    def test_2vx4_fix_dot(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar,x4_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(x4_vector,scalar))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="dot")
        print("\n".join(mmgen.generate()))

    def test_2vx4_vla_dot(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar,vla_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(vla_vector,scalar))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(scalar,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="dot")
        print("\n".join(mmgen.generate()))

    def test_4vx4_idxx4_dot(self):
        m = 4
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(scalar,x4_vector))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(x4_vector,scalar))
        c_tile = simple_ukr_tile(a_size=m//x4_vector.sd_size, b_size=n, subdims=(x4_vector,scalar))

        mmgen = dummy_mm_op(a_tile, b_tile, c_tile)
        mmgen.set_op(opstr="dot")
        print("\n".join(mmgen.generate()))
