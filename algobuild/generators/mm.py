

from abc import abstractmethod
from enum import Enum,auto

from ..components import operation
from ..components import tile,dimension_type

class loop_order:
    max_direct_dimchars = 5 # 8 # This is 40k permutations, dynamic computation is probably faster
    def __init__(self, dimchars : list[str]):
        self.dimchars = dimchars
        if len(dimchars) <= loop_order.max_direct_dimchars:
            indices = list(range(len(dimchars)))
            import itertools
            permutations = itertools.permutations(indices)
            self.directmap = {
                "".join([dimchars[i] for i in p]) : p for p in permutations
            }
        else:
            self.dimchar_indices = { d : i for i,d in enumerate(dimchars) }
    def __call__(self, order : str):
        assert len(order) == len(self.dimchars)
        assert set(order) == set(self.dimchars)
        if len(order) <= loop_order.max_direct_dimchars:
            return self.directmap[order]
        return [self.dimchar_indices[c] for c in order]

# In case either future me or someone else tries searching for the definition of these
# order2D, order3D, order4D, order5D, order6D, order7D, order8D, order9D
all_dimchars = ['m','n','k','l','o','p','r','s','t']
for i in range(3,len(all_dimchars)+1):
    dimchars = [f(c) for c in all_dimchars[:i] for f in (lambda c : c.upper(), lambda c : c)]
    name = "".join(dimchars)+"_order"
    vars()[name] = loop_order(dimchars=dimchars)
    vars()[f"order{i-1}D"] = vars()[name]

# The above does basically:
# MmNnKk_order = loop_order(dimchars=['M','m','N','n','K','k'])
# order2D      = MmNnKk_order
# ... etc


class mm_op(operation):
    def __init__(self,
                 a_tile : tile, b_tile : tile, c_tile : tile,
                 a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                 m_subidx : int, n_subidx : int, k_subidx : int,
                 opstr : str = "fma"):
        super().__init__(
                 tiles=[a_tile,b_tile,c_tile],
                 tile_offsets=[list(a_idx),list(b_idx),list(c_idx)],
                 subindices=[m_subidx, n_subidx, k_subidx],
                 opstr=opstr)
    @property
    def a_tile(self):
        return self.tiles[0]

    @property
    def b_tile(self):
        return self.tiles[1]

    @property
    def c_tile(self):
        return self.tiles[2]

    @property
    def a_idx(self):
        return self.tile_offsets[0]

    @property
    def b_idx(self):
        return self.tile_offsets[1]

    @property
    def c_idx(self):
        return self.tile_offsets[2]

    @property
    def m_subidx(self):
        return self.subindices[0]

    @property
    def n_subidx(self):
        return self.subindices[1]

    @property
    def k_subidx(self):
        return self.subindices[2]


class mm:
    def __init__(self, a : tile, b : tile, c : tile, lo : list[int] = order2D('mnkMNK'), opstr : str = "fma"):
        self.a = a
        self.b = b
        self.c = c
        self.lo = lo
        self.opstr = opstr

    @property
    def a_tile(self) -> tile:
        return self.a
    @property
    def b_tile(self) -> tile:
        return self.b
    @property
    def c_tile(self) -> tile:
        return self.c

    
    def generate(self, add_dims : list[int] = None) -> list[mm_op]:
        if None == add_dims:
            add_dims = [0]*len(self.lo)

        # TODO: Think about how to make the index order clear in the interface
        #       - is the order the same as lo or fixed?
        add_M = add_dims[3]
        add_N = add_dims[4]
        add_K = add_dims[5]
        add_m = add_dims[0]
        add_n = add_dims[1]
        add_k = add_dims[2]

        result = self.compute_subtiles(self.a, self.b, self.c,
                                       a_off=(add_M,add_K),b_off=(add_K,add_N),c_off=(add_M,add_N),
                                       m_subidx=add_m, n_subidx=add_n, k_subidx=add_k
                                       )
        return result

    def get_indices(self, tile_idx : int, dims : list[int]) -> list[int]:
        indices = [0 for d in dims]
        blocksize = 1
        for idx in self.lo:
            indices[idx] = (tile_idx // blocksize) % dims[idx]
            blocksize *= dims[idx]

        return indices

    def compute_subtiles(self,
            a_tile : tile, b_tile : tile, c_tile : tile,
            a_off : (int,int), b_off : (int,int), c_off : (int,int),
            m_subidx : int, n_subidx : int, k_subidx : int,
            ) -> list[mm_op]:

        M_small = min(a_tile.dima.size, c_tile.dima.size)
        M_big = max(a_tile.dima.size, c_tile.dima.size)
        M_factor = M_big//M_small

        N_small = min(b_tile.dimb.size, c_tile.dimb.size)
        N_big = max(b_tile.dimb.size, c_tile.dimb.size)
        N_factor = N_big//N_small

        K_small = min(a_tile.dimb.size, b_tile.dima.size)
        K_big = max(a_tile.dimb.size, b_tile.dima.size)
        K_factor = K_big//K_small

        M = M_small
        N = N_small
        K = K_small


        if not c_tile.subtiles:
            return [mm_op(
                a_tile=a_tile, b_tile=b_tile, c_tile=c_tile, 
                a_idx=a_off, b_idx=b_off, c_idx=c_off,
                m_subidx=m_subidx, n_subidx=n_subidx, k_subidx=k_subidx, 
                opstr=self.opstr
            )]

        result = []
        tile_count = M*N*K*M_factor*N_factor*K_factor
        for tile_idx in range(tile_count):
            # Some parts are already kind-of set up for multidim/tensors
            # but let's focus on GEMM for now
            indices = self.get_indices(tile_idx=tile_idx, dims=[M, M_factor, N, N_factor, K, K_factor])
            m_idx = indices[0]
            n_idx = indices[2]
            k_idx = indices[4]
            m_subidx = indices[1]
            n_subidx = indices[3]
            k_subidx = indices[5]
            #print(f"(m,n,k)=({m_idx},{n_idx},{k_idx})")


            a_absolute_idx = tuple([sum(x) for x in zip(a_off,(m_idx,k_idx))]) 
            b_absolute_idx = tuple([sum(x) for x in zip(b_off,(k_idx,n_idx))]) 
            c_absolute_idx = tuple([sum(x) for x in zip(c_off,(m_idx,n_idx))]) 

            m_subtile_idx = m_idx % c_tile.subtile_count_a
            n_subtile_idx = n_idx % c_tile.subtile_count_b
            k_subtile_idx = k_idx % a_tile.subtile_count_b

            c_subtile = c_tile.subtiles[n_subtile_idx*c_tile.subtile_count_a + m_subtile_idx]
            b_subtile = b_tile.subtiles[n_subtile_idx*b_tile.subtile_count_a + k_subtile_idx]
            a_subtile = a_tile.subtiles[k_subtile_idx*a_tile.subtile_count_a + m_subtile_idx]

            a_suboffset = tuple([idx*dim for idx,dim in zip(a_absolute_idx,
                                                            (a_subtile.dima.size, a_subtile.dimb.size))])
            b_suboffset = tuple([idx*dim for idx,dim in zip(b_absolute_idx,
                                                            (b_subtile.dima.size, b_subtile.dimb.size))])
            c_suboffset = tuple([idx*dim for idx,dim in zip(c_absolute_idx,
                                                            (c_subtile.dima.size, c_subtile.dimb.size))])


            subresult = self.compute_subtiles(
                a_tile=a_subtile, b_tile=b_subtile, c_tile=c_subtile, 
                a_off=a_suboffset, b_off=b_suboffset, c_off=c_suboffset, 
                m_subidx=m_subidx, n_subidx=n_subidx, k_subidx=k_subidx, 
            )
            result.extend(subresult)
        return result


class string_mapper:
    def __init__(self, op : str = "fma"):
        self.op = op
    def __call__(self, ops : list[mm_op]) -> list[str]:

        result = []
        idxstr = lambda dima,dimb,sidx : "" if dima.size==dimb.size else f".el[{sidx}]" if dima.size > dimb.size else f"+{sidx}"
        vlensuf = lambda d : "*VLEN" if d.dt == dimension_type.vla else ""
        for op in ops:

            a_idx_str = (f"({op.a_idx[0]}{vlensuf(op.a_tile.dima)}{idxstr(op.a_tile.dima,op.c_tile.dima,op.m_subidx)},"
                          f"{op.a_idx[1]}{vlensuf(op.a_tile.dimb)}{idxstr(op.a_tile.dimb,op.b_tile.dima,op.k_subidx)})")
            b_idx_str = (f"({op.b_idx[0]}{vlensuf(op.b_tile.dima)}{idxstr(op.b_tile.dima,op.a_tile.dimb,op.k_subidx)},"
                          f"{op.b_idx[1]}{vlensuf(op.b_tile.dimb)}{idxstr(op.b_tile.dimb,op.c_tile.dimb,op.n_subidx)})")
            c_idx_str = (f"({op.c_idx[0]}{vlensuf(op.c_tile.dima)}{idxstr(op.c_tile.dima,op.a_tile.dima,op.m_subidx)},"
                          f"{op.c_idx[1]}{vlensuf(op.c_tile.dimb)}{idxstr(op.c_tile.dimb,op.b_tile.dimb,op.n_subidx)})")
            result.append(
                f"C{c_idx_str} <- {self.op}(A{a_idx_str},B{b_idx_str},C{c_idx_str})"
            )
        return result
