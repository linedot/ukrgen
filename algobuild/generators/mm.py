# annotation for self type
from __future__ import annotations

from abc import abstractmethod
from enum import Enum,auto

class storage_type(Enum):
    register = auto()
    memory = auto()

class dimension_type(Enum):
    fixed = auto()
    vla = auto()
    # I think we don't need this, and fixed fits the functionality
    # subtile = auto()

class dimension_properties:
    def __init__(self, dt : dimension_type, size : int, sdt : dimension_type, sd_size : int):
        self.dt = dt
        self.size = size
        self.sdt = sdt
        self.sd_size = sd_size


class tile:
    def __init__(self, 
                 dima : dimension_properties, # m for c, m for a, k for b
                 dimb : dimension_properties, # n for c, k for a, n for b
                 # 3D tiles? 3D registers? Tensors? :D maybe in the future
                 # dims : list[dimension_properties]
                 # Thinking of tiling with differently-sized kernels in 2D...
                 subtiles : list[tile] = None,
                 subtile_count_a : int = None,
                 subtile_count_b : int = None,
                 # ND?
                 # subtile_counts : list[int] = None,
                 stype : storage_type = storage_type.register):
        self.storage_type = storage_type
        self.dima = dima
        self.dimb = dimb
        self.subtiles = subtiles
        self.subtile_count_a = subtile_count_a
        self.subtile_count_b = subtile_count_b

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


class mm_op:
    def __init__(self, a : tile, b : tile, c : tile, lo : list[int] = order2D('mnkMNK'), reorder_c : bool = False):
        self.a = a
        self.b = b
        self.c = c
        self.lo = lo
        self.reorder_c = reorder_c

        self.additional_init()

    @property
    def a_tile(self) -> tile:
        return self.a
    @property
    def b_tile(self) -> tile:
        return self.b
    @property
    def c_tile(self) -> tile:
        return self.c

    def additional_init(self):
        pass
    
    def generate(self) -> list[str]:

        result = self.compute_subtiles(self.a, self.b, self.c,
                                       a_off=(0,0),b_off=(0,0),c_off=(0,0),
                                       m_subidx=0, n_subidx=0, k_subidx=0 
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
            m_subidx : int, n_subidx : int, k_subidx : int
            ) -> list[str]:


        #M = a_tile.dima.size
        #assert M == c_tile.dima.size, "M dim not matching between A and C"
        #N = b_tile.dimb.size
        #assert N == c_tile.dimb.size, "N dim not matching between B and C"
        #K = a_tile.dimb.size
        #assert K == b_tile.dima.size, "K dim not matching between A and B"

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
            return self.computation_atom(
                a_tile=a_tile, b_tile=b_tile, c_tile=c_tile, 
                a_idx=a_off, b_idx=b_off, c_idx=c_off,
                m_subidx=m_subidx, n_subidx=n_subidx, k_subidx=k_subidx, 
            )

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

    @abstractmethod
    def computation_atom(self,
                         a_tile : tile, b_tile : tile, c_tile : tile,
                         a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                         m_subidx : int, n_subidx : int, k_subidx : int) -> list[str]:
        raise NotImplementedError("abstract mm_op does nothing")


class dummy_mm_op(mm_op):
    def additional_init(self):
        self.op = "op"
    def set_op(self, opstr : str):
        self.op = opstr
    def computation_atom(self,
                         a_tile : tile, b_tile : tile, c_tile : tile,
                         a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                         m_subidx : int, n_subidx : int, k_subidx : int) -> list[str]:

        idxstr = lambda dima,dimb,sidx : "" if dima.size==dimb.size else f".el[{sidx}]" if dima.size > dimb.size else f"+{sidx}"
        vlensuf = lambda d : "*VLEN" if d.dt == dimension_type.vla else ""
        a_idx_str = (f"({a_idx[0]}{vlensuf(a_tile.dima)}{idxstr(a_tile.dima,c_tile.dima,m_subidx)},"
                      f"{a_idx[1]}{vlensuf(a_tile.dimb)}{idxstr(a_tile.dimb,b_tile.dima,k_subidx)})")
        b_idx_str = (f"({b_idx[0]}{vlensuf(b_tile.dima)}{idxstr(b_tile.dima,a_tile.dimb,k_subidx)},"
                      f"{b_idx[1]}{vlensuf(b_tile.dimb)}{idxstr(b_tile.dimb,c_tile.dimb,n_subidx)})")
        c_idx_str = (f"({c_idx[0]}{vlensuf(c_tile.dima)}{idxstr(c_tile.dima,a_tile.dima,m_subidx)},"
                      f"{c_idx[1]}{vlensuf(c_tile.dimb)}{idxstr(c_tile.dimb,b_tile.dimb,n_subidx)})")
        return [
            f"C{c_idx_str} <- {self.op}(A{a_idx_str},B{b_idx_str}) + C{c_idx_str}"
        ]

class tile_offset_mapper:
    @abstractmethod
    def __call__(
            tile : tile,
            idx : (int,int)) -> int:
# I don't think we need the subindex for data origins
#            subidx : int) -> int:
        raise NotImplementedError("tried calling abstract tile offset mapper")


class resource_mm_op(mm_op):
    def __init__(self,
                 res_counts : list[int],
                 res_steps : list[int],
                 addr_counts : list[int],
                 addr_offset_ranges : list[list[tuple[int,int]]],
                 addr_starts : list[list[int]],
                 preload_counts : list[int],
                 mappers : list[tile_offset_mapper], 
                 a : tile, b : tile, c : tile,
                 op : str = 'fma',
                 lo : list[int] = order2D('mnkMNK'),
                 reorder_c : bool = False):
        super(resource_mm_op, self).__init__(
                a=a, b=b, c=c,
                lo=lo, reorder_c=reorder_c
                )
        for rc,pc in zip(res_counts,preload_counts):
            assert pc <= rc, "Can't preload more than max. specified number of resources"

        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.mappers = mappers
        self.op = op

        self.res_indices = [0]*len(res_counts)
        self.res_subindices = [0]*len(res_counts)

        self.addr_counts = addr_counts
        self.addr_indices = [0]*len(addr_counts)
        self.addr_offset_ranges = addr_offset_ranges

        # For now: -1 = invalid data, replace origin, schedule load
        #          -2 = data preloaded, replace origin, but don't schedule a load
        # TODO: revisit with not-magic-number based approach?
        self.cdos = [
            { i : -2 if i < pre_count else -1 for i in range(res_count)}\
                    for res_count,pre_count,idx,subidx in\
                    zip(res_counts,preload_counts,
                        self.res_indices,self.res_subindices)
        ]
        # Offsets to base address corresponding to absolute address values 
        # currently stored in the address registers
        # -2 = assume correct value in addr, replace tracked value, but don't
        #      schedule update operation
        self.addr_reg_offsets = [
            { i : -2 if i < pre_count else starts[i] for i in range(addr_count)}\
                    for pre_count,starts,addr_count in\
                    zip(preload_counts,addr_starts,addr_counts)
        ]
        # ABC data offsets to addresses currently stored in the address registers
        self.addr_offsets = [0]*len(res_counts)

    def is_in_range(self, current_offset : int, target_offset : int, offset_range : tuple[int,int]) -> bool:
        difference = target_offset - current_offset

        return (difference >= offset_range[0]) and (difference <= offset_range[1])

    def resolve_data(self, res_idx : int, toff : int, rtype_idx : int, vla : bool = False) -> list[str]:
        # check if the required data is already in the resource
        reg_chars = ["a","b","c"]
        corg = self.cdos[rtype_idx][res_idx]
        if corg == -2:
            self.cdos[rtype_idx][res_idx] = toff
            corg = toff
        if corg == toff:
            return []

        vlenstr = ""
        if vla:
            vlenstr = "*VLEN"

        result = []

        caoffs = self.addr_reg_offsets[rtype_idx]

        addr_idx_to_use = -1
        while -1 == addr_idx_to_use:
            # check if we can address the data with one of the address registers + immediate offset
            for addr_idx in range(self.addr_counts[rtype_idx]):
                caoff = caoffs[addr_idx]
                offset_range = self.addr_offset_ranges[rtype_idx][addr_idx]
                if (caoff == -2):
                    addr_idx_to_use = addr_idx
                    self.addr_offsets[rtype_idx] = 0
                    self.addr_reg_offsets[rtype_idx][addr_idx] = toff
                    break
                if self.is_in_range(current_offset=caoff,
                               target_offset=toff,
                               offset_range=offset_range):
                    addr_idx_to_use = addr_idx
                    self.addr_offsets[rtype_idx] = toff-caoff
                    break
            if -1 != addr_idx_to_use:
                break

            # First address register
            addr_idx_to_increment = 0
            caoff_max = self.addr_reg_offsets[rtype_idx][addr_idx_to_increment]
            # if none of the registers are in range, increment the addr register with the highest value below target
            for addr_idx in range(self.addr_counts[rtype_idx]):
                caoff = self.addr_reg_offsets[rtype_idx][addr_idx]
                if caoff_max < caoff and caoff < toff:
                    addr_idx_to_increment = addr_idx
                    caoff_max = caoff

            # add first value of the ranges (i.e. if range is (-16,15), subtract 16 so that
            # target offset can be reached with the lowest offset value in the range)
            add_value = toff - caoff_max + self.addr_offset_ranges[rtype_idx][addr_idx_to_increment][0]
            ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_increment}"
            result.append(f"{ar_str} <- {ar_str} + {add_value}{vlenstr}")
            self.addr_reg_offsets[rtype_idx][addr_idx_to_increment] += add_value
        
        
        ar_str = f"{reg_chars[rtype_idx]}a{addr_idx_to_use}"
        res_str = f"{reg_chars[rtype_idx]}{res_idx}"

        self.cdos[rtype_idx][res_idx] = toff
        offpref = "o"
        if vla:
            offpref = ""
        result.append(f"{res_str} <- LOAD {ar_str} + {offpref}{self.addr_offsets[rtype_idx]}{vlenstr}")

        return result

        
    def computation_atom(self,
                         a_tile : tile, b_tile : tile, c_tile : tile,
                         a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                         m_subidx : int, n_subidx : int, k_subidx : int) -> list[str]:

        target_offsets = [ mapper(tile,idx) for  mapper,tile,idx in\
                zip(self.mappers,[a_tile,b_tile,c_tile],[a_idx,b_idx,c_idx])]

        result = []
        res_indices = []

        off_is_vla = [(t.dima.dt == dimension_type.vla) or (t.dimb.dt == dimension_type.vla) for t in [a_tile,b_tile,c_tile] ]

        for i,(toff,dos,res_count) in enumerate(
                zip(target_offsets,self.cdos,self.res_counts)):
            res_idx = -1

            # check if any of the registers have the data
            for j in range(res_count):
                if (toff == dos[j]) or (-2 == dos[j]):
                    res_idx = j
                    break

            # use the current index for this input, use the next next time
            if res_idx == -1:
                res_idx = self.res_indices[i]
                self.res_indices[i] = (res_idx+self.res_steps[i]) % self.res_counts[i]

            res_indices.append(res_idx)
            #print(f"Data change o{corg} -> o{toff*ofac} for dreg {suf}{reg} required")
            result.extend(self.resolve_data(res_idx=res_idx, toff=toff, rtype_idx=i, vla=off_is_vla[i]))

        areg = res_indices[0]
        breg = res_indices[1]
        creg = res_indices[2]
        result.append(f"c{creg} <- {self.op}(a{areg},b{breg},c{creg})")
        return result

class dataflow_mm_op(mm_op):
    def __init__(self,
                 a : tile, b : tile, c : tile,
                 lo : list[int] = order2D('mnkMNK'),
                 reorder_c : bool = False):
        super(resource_mm_op, self).__init__(
                a=a, b=b, c=c,
                lo=lo, reorder_c=reorder_c
                )
    def computation_atom(self,
                         a_tile : tile, b_tile : tile, c_tile : tile,
                         a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                         m_subidx : int, n_subidx : int, k_subidx : int) -> list[str]:
        # Won't be quite as easy as I thought, as you kinda have to add nodes to
        # a graph and then finalize it and the idea of a computation_atom on it's
        # own actually already implies an instruction based architecture
        pass
